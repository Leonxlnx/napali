// ---------------------------------------------------------------- vegetation + rock geometry (signed distance fields)
// Every tree, palm and bush is an individual object, instanced on jittered grids that a
// generation pass fills from the terrain / vegetation masks. Rock faces get true 3D relief.
float gPix=.01;          // pixel footprint (m) at the current sample
bool gVeg=true;          // evaluate trees / bushes at this sample
const float ENV=32.;     // vegetation envelope above the ground (m)

float sdEll(vec3 q, vec3 r){ float k0=length(q/r), k1=length(q/(r*r)); return k0*(k0-1.)/max(k1,1e-5); }
float sdCap(vec3 p, vec3 a, vec3 b, float r){ vec3 pa=p-a, ba=b-a; float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.); return length(pa-ba*h)-r; }
vec2 tJit(vec4 T){ return fract(T.xy*.5)*2.; }   // cell jitter (the integer part carries a lean: x = direction /16, y = amount)
vec2 tLean(vec4 T){ float k=floor(T.x*.5), m=floor(T.y*.5); float a=k*.3926991; return m>0.?vec2(cos(a),sin(a))*m:vec2(0); }
vec4 cellTex(sampler2D s, vec2 c){ ivec2 sz=textureSize(s,0); return texelFetch(s,clamp(ivec2(c),ivec2(0),sz-1),0); }

// ---- broadleaf crowns: crown envelope -> leaf clusters (~1-1.5 m) -> leaf sprays (~0.5 m) -> individual leaves.
// Every level fades out before it gets smaller than ~2 pixels (band-limited, no sparkle).
float gEpsM=0.;         // (r5) hit tolerance of the current primary march step (0 elsewhere): cheap-bound early-outs keep their value above it
float lodF(float size){ return clamp(size/max(gPix,1e-4)*.5-1.,0.,1.); }   // 1 when size >= 4 px, 0 at <= 2 px
// shading hints written by the crown that won the last treesSD / bushSD evaluation
float gClId, gDe, gSpr, gLeafId, gLeafW, gInner, gCarpIn; vec3 gClOff; vec4 gCrC;   // gCrC: crown centre (xyz) and size R of the crown that won
float gCarpHit=0.;   // (r9 side) gCarpIn of the primary hit (saved by march: the shading no longer re-evaluates terrSD for it)
float tClId, tDe, tSpr, tLeafId, tLeafW, tInner; vec3 tClOff;   // scratch (current candidate)
// leaf sprays: one rounded bump per spray (worley cells), x = dome 0..1
float sprayDome(vec3 p, float cs){
  vec3 q=p/cs, n=floor(q), f=fract(q);
  float d1=8.;
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 r=o+.2+.6*hash33(c)-f;
    d1=min(d1,dot(r,r));
  }
  float v=max(1.-d1*2.2,0.);
  return v*v*(3.-2.*v);                                      // soft foot: no crease ring around each dome
}
// leaf clusters / sprays as shading-only bumps (below the geometric LOD): x = dome (0 in a gap .. 1 on top),
// yzw = gradient of the dome height (1/m)
vec4 domeG(vec3 p, float cs){
  vec3 q=p/cs, n=floor(q), f=fract(q);
  float d1=8.; vec3 rb=vec3(0);
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 r=o+.2+.6*hash33(c+5.3)-f;
    float d=dot(r,r);
    if(d<d1){ d1=d; rb=r; }
  }
  float s=max(1.-d1*2.2,0.);
  return vec4(s*s*(3.-2.*s),s>0.?6.*s*(1.-s)*4.4*rb/cs:vec3(0));
}
// individual leaves: one elliptic blade per jittered cell (size ls), facing up / outward, tip drooping.
// dbase = distance to the foliage volume at p: every leaf is clipped at its own depth -> ragged, leafy outline
float leavesSD(vec3 p, float ls, float seed, float dbase){
  vec3 g=p/ls, n=floor(g), f=fract(g);
  float d=1e9;
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 h=hash33(c+seed), h2=hash33(c*1.71+seed+3.1);
    if(h2.z>.88) continue;                                     // a few empty cells: gaps
    vec3 q=(f-o-.25-.5*h)*ls;
    vec3 nn=normalize(h2*2.-1.+vec3(0,.8,0));
    float s=nn.z>=0.?1.:-1., aa=-1./(s+nn.z), bb=nn.x*nn.y*aa;
    vec3 t1=vec3(1.+s*nn.x*nn.x*aa,s*bb,-s*nn.x), t2=vec3(bb,s+nn.y*nn.y*aa,-nn.y);
    float an=h.z*6.2831853;
    vec3 a=cos(an)*t1+sin(an)*t2, b=cross(nn,a);
    float L=ls*(.38+.24*h.x);                                  // half length
    float x=dot(q,a), y=dot(q,b), z=dot(q,nn)+(.15+.4*h2.y)*x*x/ls+.25*y*y/ls;   // droops along, cups across the midrib
    float xc=clamp(x/L,-1.,1.);
    float e=max(abs(x)-L,(abs(y)-L*(.3+.12*h.y)*(1.-xc*xc))*.75);              // pointed lance / ovate blade
    float dl=max(max(e,abs(z)-.004-gPix*.45),dbase-(.25+1.8*h2.x)*min(ls,.2)-.5*pow(h.y,6.));   // a few leaves sit deep inside: leafy interior
    if(dl<d){ d=dl; tLeafId=h2.y; }
  }
  return d;
}
// multi-scale leafy relief for a foliage surface: leaf clusters (~1.1 m) and sprays (~0.45 m), band-limited.
// tSpr = crevice occlusion factor (mean 1: dark in the gaps between clusters / sprays, bright on their tops)
// (r5) leaf clumps of very different sizes: per-cell size .55-1 and some empty cells, overlapping (max of the domes),
// with a soft foot -> ragged clumps, not a field of identical round pebbles
float sprayDomeV(vec3 p, float cs){
  vec3 q=p/cs, n=floor(q), f=fract(q);
  float v=0.;
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 h=hash33(c);
    vec3 r=o+.2+.6*h-f;
    float sz=fract(h.x*7.13+h.z*3.7);
    if(sz<.12) continue;                                        // gap
    float kr=mix(.55,1.,sz);
    float s=clamp((1.-dot(r,r)*2.2/(kr*kr))*1.25,0.,1.);
    v=max(v,s*s*(3.-2.*s)*mix(.65,1.,fract(h.y*5.3)));
  }
  return v;
}
float foliageRelief(vec3 p, float fine){
  float f1=lodF(2.2), f2=lodF(1.)*clamp((.3-gPix)*10.,0.,1.), f3=lodF(.45)*fine;   // f2 done by gPix .3 (keeps the shadow LOD pass cheap)
  float r=0., ao=1.;
  if(f1+f2>0.) p+=noised3(p*.45+2.).yzw*.55;                   // warped: irregular leaf clusters, not a sheet of round bubbles
  if(f1>0.){ float s=sprayDomeV(p+7.1,2.2); r+=(s-.45)*.6*f1; ao*=mix(1.,(.66+.4*s)/.86,f1); }   // (r5: softer crevice rims: no dark outline round every 2 m clump)
  if(f2>0.){ float s=sprayDomeV(p,1.); r+=(s-.45)*.3*f2; ao*=mix(1.,(.62+.38*s)/.81,f2); }
  if(f3>0.){ float s=sprayDome(p+3.7,.45); r+=(s-.45)*.13*f3; ao*=mix(1.,(.62+.42*s)/.80,f3); }
  tSpr=ao;
  return r;
}
// leaf clusters on a crown lobe: ~1 m rosettes with deep, dark gaps between them, on a 2.2 m undulation (tSpr = gap occlusion)
float crownRelief(vec3 p){
  float f1=lodF(2.2), f2=lodF(1.05)*clamp((.3-gPix)*10.,0.,1.);
  float r=0., ao=1.;
  vec3 wq=f1>0.?noised3(p*.8).yzw*.3:vec3(0);
  if(f1>0.){ float s=sprayDomeV(p+7.1+wq*1.5,2.2); r+=(s-.45)*.62*f1; ao*=mix(1.,(.5+.6*s)/.77,f1); }   // (r5: clusters of varied size with gaps, warped: ragged crown outlines with protruding clumps, not cauliflower bumps)
  if(f2>0.){ float s=sprayDome(p+wq,1.05); r+=(s-.55)*.8*f2; ao*=mix(1.,(.42+.75*s)/.8,f2); }   // warped: irregular clusters, not round bubbles
  tSpr=ao;
  return r;
}
// ragged, lobed crown envelope (crown-local q): a few big lobes + smaller bulges, no perfect ellipsoids
float crownEnv(vec3 q, vec3 rad, float R, float seed, float rag){
  float de=sdEll(q,rad);
  float n1=noise3(q*(1.5/R)+seed*9.)*.5+.5, n2=noise3(q*(3.4/R)+seed*5.+2.)*.5+.5;
  return de-R*rag*(1.+.7*smoothstep(.3,1.,gPix))*(.62*n1+.38*n2-.35);   // from afar: more ragged, cauliflower outlines
}
// leaves at every distance: the blade size steps up in octaves (x2) so a leaf never gets smaller than ~2.5 px;
// far away a 'leaf' stands for a tuft, which keeps the leafy light / gap structure instead of smooth bumps.
// returns the blended distance; maxOct limits the coarsest octave
float leafOct(vec3 p, float leafS, float seed, float dbase, float maxOct){
  float kf=clamp(log2(max(3.6*gPix/leafS,1.)),0.,maxOct);   // a leaf never smaller than ~3.6 px: leafy, not grainy
  float k0=floor(kf), fk=smoothstep(.72,1.,fract(kf));
  if(kf>=maxOct) fk=0.;
  float s0=exp2(k0);
  float shell=2.05*min(leafS*s0*(fk>0.?2.:1.),.2)+leafS*s0*(fk>0.?2.:1.)*.6+.5;
  if(dbase>shell+.05+2.*gEpsM){ tLeafId=.5; return dbase-shell; }   // far from the foliage shell: cheap bound (r5: the value returned at the threshold, x .6 in crownSD, stays above the current march hit tolerance gEpsM, or the bound itself is hit as a false dotted hull around every crown at grazing angles)
  float d0=leavesSD(p,leafS*s0,seed+k0*13.,dbase);
  float id0=tLeafId;
  if(fk>0.){ float d1=leavesSD(p,leafS*s0*2.,seed+k0*13.+13.,dbase); d0=mix(d0,d1,fk); if(fk<.5) tLeafId=id0; }
  return d0;
}
// leaf clusters on a jittered grid inside the crown envelope
// the cluster lattice is rotated / tilted per tree, so clusters never line up in horizontal rows (no terraced hedges)
mat3 latticeRot(float seed){
  float a=seed*6.2831, b=.45+.35*fract(seed*7.7), c=fract(seed*3.3)*6.2831;
  float ca=cos(a), sa=sin(a), cb=cos(b), sb=sin(b), cc=cos(c), sc=sin(c);
  return mat3(ca,0.,-sa, 0.,1.,0., sa,0.,ca)*mat3(1.,0.,0., 0.,cb,sb, 0.,-sb,cb)*mat3(cc,0.,-sc, 0.,1.,0., sc,0.,cc);
}
float clusterField(vec3 q, float cs, float seed, vec3 rad, float R, float rag, float rs){
  mat3 M=latticeRot(seed);
  vec3 g=(M*q)/cs, n=floor(g), f=fract(g);
  float best=1e9;
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 h=hash33(c+seed*17.3);
    vec3 cc=((c+.1+.8*h)*cs)*M;                               // cluster centre (crown-local, back from the lattice frame)
    float env=sdEll(cc,rad*(.74+.45*rag*h.z));
    if(env>0.) continue;
    float r=cs*(rs+.24*h.x)*(1.-.3*smoothstep(-.3*R,0.,env));
    vec3 dq=q-cc; dq.y*=1.35;                                 // flattish clusters (leaves at branch tips)
    float d=length(dq)-r;
    if(d<best){ best=d; tClId=h.y; tClOff=dq/r; }
  }
  return best/1.35;
}
// R = crown size (m); leafS = leaf length (m); rag = raggedness of the outline
// Crown v3: K leaf masses at the branch tips, set on the outer shell of the crown volume (a crown is hollow: dark
// limbs and a sparse inner core show through the gaps between the masses), joined softly at their bases.
// Near: every mass carries leaf clusters (2.2 m / 1 m) and real leaves. Far: masses stay until ~5 px, then the lobed envelope.
float tLimb=0.;
float crownSD(vec3 p, vec3 cen, vec3 rad, float seed, float R, float leafS, float rag){
  vec3 q=p-cen;
  float de=sdEll(q,rad);
  tDe=de/R; tSpr=1.; tClId=seed; tClOff=q/rad; tLeafW=0.; tInner=0.; tLimb=0.;
  if(de>R*.6+1.2*gEpsM) return de-R*.5;   // (r5: threshold above the hit tolerance: returning ~.1R there made a thin false shell around every crown, hit only at grazing angles = the dotted 'hoops' on skylines)
  float env=crownEnv(q,rad,R,seed,rag);
  vec3 offE=normalize(q/rad+vec3(0,.6,0));
  float f2=clamp(R*.4/gPix/1.2-1.,0.,1.);                     // masses full at ~3 px each (seen from the air too); below, one lobed dome
  if(f2<=0.){ tClOff=offE; tDe=0.; return env*.75; }
  float K=floor(7.+4.*fract(seed*5.13));                         // 7..10 branch-tip masses
  float a0=seed*6.2831853;
  float d=1e9, d2=1e9, bi=0.; vec3 bo=vec3(0,1,0);
  float dlb=1e9;
  bool limbs=gPix<.2&&uDbgM!=50;
  vec3 tb=vec3(0.,-rad.y*.6,0.);                                 // top of the trunk (crown base)
  for(int i=uZero;i<10;i++){
    float fi=float(i); if(fi>=K) break;
    vec3 h=hash33(vec3(seed*53.17+fi*1.731,fi*7.13+seed*3.1,seed*11.7-fi));
    float az=a0+(fi+.8*h.x)*6.2831853/K;
    float el=fi<.5?1.4:mix(-.15,1.15,h.y);                        // elevation of the branch tip: low wide masses .. upper masses
    float rho=fi<.5?.3:mix(.38,.72,h.z);
    float hs=fract(h.x*7.7+h.z*3.1);
    float s=R*mix(.18,.46,hs*hs*(3.-2.*hs))*(fi<.5?1.1:1.);
    if(fract(h.y*13.1)<.4&&fi>.5){ rho=mix(.84,1.,h.z); s*=.55; }   // small masses sticking far out: ragged outline (r5: more of them, further out)
    vec3 c=rad*rho*vec3(cos(el)*cos(az),sin(el),cos(el)*sin(az));
    vec3 sr=vec3(s,s*mix(.68,.92,h.y),s);                          // flattish: leaves in horizontal sprays (layered crowns) ((r10 canopy) .55-.8 -> .68-.92: rounded masses, not stacked pads)
    vec3 dq=q-c;
    float dm=sdEll(dq,sr);
    if(dm<d){ d2=d; d=dm; bi=fi; bo=dq/sr; } else d2=min(d2,dm);
    if(limbs) dlb=min(dlb,sdCap(q,tb,c*.85,R*.022+.035));
  }
  float kk=R*.09, hh=max(kk-abs(d-d2),0.)/kk; d-=hh*hh*kk*.25;   // masses join softly at their bases, deep clefts between them (r5: less webbing: see-through gaps)
  d-=R*rag*.3*(noise3(q*(2.4/R)+seed*11.)*.5+.25);               // ragged outlines
  float inn=0.;
  d=mix(env,d,f2);
  tClId=fract(bi*.618+seed*3.7); tClOff=mix(offE,normalize(bo+vec3(0,.2,0)),f2); tDe=env/R*f2;
  if(gPix>.3||uDbgM==52){ tInner=inn; return d*.72; }
  float dr=d-crownRelief(p+seed*31.);                           // leaf clusters with dark gaps on the masses
  tLeafW=1.;
  float dl=leafOct(p,leafS,seed*57.,dr-.06,3.), fb=dr+.6;     // leaves; the dense interior .6 m below shows through the gaps
  tInner=max(max(smoothstep(-.03,.03,dl-fb),smoothstep(.0,.5,-dr)),inn);
  float dm=uDbgM==21?dr:min(dl,fb);
  if(limbs&&dlb<dm*.6){ tLimb=1.; return dlb; }
  return dm*.6;
}
// tree species (from the per-tree seed): 0 dark dome (mango, java plum), 1 mid-green dome, 2 umbrella (albizia, monkeypod),
// 3 pale open crown (kukui), 4 yellow-green tangle (hau, guava)
float treeType(float seed){ float k=fract(seed*7.31); return k<.32?0.:k<.6?1.:k<.75?2.:k<.83?3.:4.; }   // kukui (3) rarer: few pale crowns
void treeGeo(vec2 c, vec4 T, out vec3 base, out vec3 cen, out vec3 rad, out float sp, out float seed){
  float R=mod(T.w,100.); sp=floor(T.w/100.+.01);
  seed=hash12(c*1.37+.5);
  vec2 bxz=uTG.xy+(c+tJit(T))*uTG.z;
  base=vec3(bxz.x,T.z,bxz.y);
  vec2 ln=tLean(T);
  float ty=treeType(seed), h1=hash12(c+3.3), h2=hash12(c+7.1), h3=hash12(c+9.7);
  float th=R*(.32+.55*h1);                                       // crowns reach down: a closed canopy from above, short trunks under it
  rad=vec3(R*(.75+.3*seed),R*(.6+.3*h2),R*(.75+.3*h3));       // crowns a bit wider than tall
  if(ty==2.){ float wu=mix(1.2,1.04,smoothstep(4.,6.5,R)); rad*=vec3(wu,.92,wu); th=R*(.42+.3*h1); }        // umbrella: wide flat crown on a taller trunk ((r10 canopy) 1.3-1.1 x .8 / .55R -> 1.2-1.04 x .92 / .42R: no pancake discs on sticks on the v4 / v5 skylines)
  if(ty==4.){ rad*=vec3(1.1,.8,1.1); th=R*.05; }                // low tangle down to the ground
  if(sp>1.5) th+=R*.4;                                           // emergent
  th+=max(noise3(vec3(bxz*.016,3.7))+.25*noise3(vec3(bxz*.05,1.3)),-.3)*4.5;   // groves of tall trees and lower gaps: big lit / shaded canopy masses
  if(ln.x!=0.||ln.y!=0.) th*=.55;                                // beach trees: low crowns leaning out over the sand
  cen=base+vec3((hash12(c+1.9)-.5)*.35*R+ln.x*.4,th+rad.y,(hash12(c+4.2)-.5)*.35*R+ln.y*.4);
}
// ---- coconut palm: slender curved trunk + ~22 arching / drooping pinnate fronds in two whorls (polar repetition).
// Fronds: parabolic rachis; leaflets every ~10 cm of arc length hang down / outward in a V and curl at the tips.
vec4 gFr, tFr, gFrW, gPalmC, tPalmC;   // gPalmC: crown top (xyz) and frond length (w) of the palm that won        // frond hints: x = position along the frond (0..1), y = frond hash, z = leaflet hash, w = leaflet across (0 rachis .. 1 tip)
float asinh_(float x){ return log(x+sqrt(x*x+1.)); }
float arcF(float u){ return u*sqrt(1.+u*u)+asinh_(u); }
float frondSD(vec3 q, float a, float k, float off, float seed, float fl, float tier, float nb){
  float sec=6.2831853/k;
  float xs=(a+off)/sec, ai=floor(xs+.5), sd=xs>ai?1.:-1.;
  if(nb>.5) ai+=sd;                                     // the neighbouring frond: fronds overlap freely (no pie-slice fans)
  float h1=hash11(ai*1.7+seed*13.+tier*5.);
  if(tier<1.5&&fract(h1*31.7)<.12){                    // a few fronds missing (irregular crowns): take the frond on the other side
    if(nb>.5) return 1.5+gPix;
    ai-=sd; h1=hash11(ai*1.7+seed*13.+tier*5.);
    if(fract(h1*31.7)<.12) return 1.5+gPix;
  }
  float aa=a+off-ai*sec;
  float h2=hash11(ai*3.1+seed*7.+tier*3.), h3=hash11(ai*5.3+seed*3.+tier);
  aa+=(h3-.5)*sec*.8;                                   // irregular azimuths
  float rxz=length(q.xz);
  float gm=min(gPix*2.,.45);                            // (r5) early-out values stay above the hit tolerance at mid range too (the 2.2 m / 1 m margins keep them conservative)
  { float lt=abs(rxz*sin(aa))-.42*fl-2.2; if(lt>0.) return lt*.8+.35+gm; }   // far to the side of this frond (incl. its sideways bend): cheap early out.
  // NB every early return must stay well above the march epsilon at its threshold, or its boundary becomes a false surface
  // tier 0: spreading, 1: old drooping, 2: young upright fronds in the centre
  float up=tier<.5?mix(.3,1.2,h1):tier<1.5?mix(-.25,.35,h1):mix(1.2,2.2,h1);
  float dr=tier<.5?mix(.45,1.,h2):tier<1.5?mix(.7,1.2,h2):mix(.35,.8,h2);   // upper fronds rise and arch out, lower ones hang
  float flen=fl*(tier<.5?.85+.35*h2:tier<1.5?.75+.3*h3:.45+.25*h3);
  // horizontal reach from the arc length: steep fronds are not longer than flat ones
  float Lh=flen*4.*dr/(arcF(up)-arcF(up-2.*dr));
  float b=dr/Lh;
  float fx=rxz*cos(aa), fz=rxz*sin(aa);
  float x=clamp(fx,0.,Lh);
  fz-=(h3-.5)*.7*x*x/Lh;                                // fronds also bend sideways
  if(abs(fz)>2.4) return (abs(fz)-1.75)*.8+gm;           // beyond the longest leaflet: cheap bound (>= .52 at the threshold)
  float Y=up*x-b*x*x, Yp=up-2.*b*x;
  float t=x/Lh;
  float sArc=(arcF(up)-arcF(Yp))/(4.*b);              // arc length along the rachis
  float lat=abs(fz);
  float gs=inversesqrt(1.+Yp*Yp);
  // rachis (midrib)
  float rach=length(vec2((q.y-Y)*gs,fz))-.04*(1.-t*.7);
  rach=max(rach,max(fx-Lh,-fx-.2));
  // leaflets: longest mid-frond; near the crown they rise in a shallow V, toward the tip they hang almost straight down
  float lmax=mix(.9,1.2,h3)*(.2+.8*sin(PI*clamp(t*1.04+.03,0.,1.)))*(tier>1.5?.75:1.);
  float tV=mix(.22,.5,h1)+1.3*t*t+(tier>.5&&tier<1.5?.3:0.);   // shallow V near the crown (flat feathery blades from above), drooping toward the tip
  // leaflets (~8.5 cm apart, swept forward ~50 deg); each has its own hang angle and length, a few are torn out.
  // Wide at the rachis, tapering to the tip: a full blade near the midrib, a see-through feathery fringe outside.
  // Spacing never drops below ~4 px (it follows the pixel size); the fringe gaps stay wider than the march epsilon
  // (no hit-tolerance growth on fronds, see march) and the vertical fringe bound below keeps rays from crawling under
  // the crown, so no 'curtains'.
  float pg=max(gPix,1e-4);
  const float fw=1.2;
  float sp=max(.085,pg*5.2);                           // real leaflet spacing (~8.5 cm) close up; further away >= ~3.3 px between strands across (.64*sp; r4: 2.7 px = sub-pixel comb, 4 px = fat paddles)
  float sl=sArc-lat*(fw+.35*sin(sArc*2.3+ai*1.7+seed*9.)), li=floor(sl/sp+.5), ls=(sl-li*sp)*.64;   // .64 = 1/sqrt(1+fw^2): across the strand
  float hl=hash11(li*7.3+ai*1.3+seed*5.+tier*9.+floor(sp*40.)+step(0.,fz)*31.);
  float hl2=fract(hl*17.31), hl3=fract(hl*53.7);
  float plim=clamp(.085/(pg*5.2),0.,1.);              // < 1: spacing held at the pixel limit
  ls+=(fract(hl*91.3)-.5)*sp*mix(.1,.3,plim*plim);    // strands irregularly spaced (no regular comb); less jitter at the pixel limit (no sub-pixel gaps)
  float tVl=tV*(.55+.9*hl2)+sign(fz)*(h2-.5)*1.2*t;   // every strand hangs at its own angle (neighbours overlap in depth: no comb); the frond rolls along its length
  float lw=lmax*(.75+.4*hl3)*step(.07,hl)/sqrt(1.+tVl*tVl);   // strand of true length ~lmax: steep strands reach less far out (a few missing / torn)
  float u=clamp(lat/max(lw,.05),0.,1.);
  float cu=(.3+.4*hl2)/max(mix(lw,lmax,smoothstep(.4,.6,pg)),.2);
  float yl=Y-lat*tVl-lat*lat*cu;                       // strands curl down toward their tips
  float gl=inversesqrt(1.+Yp*Yp+pow(abs(tVl)+2.*lat*cu,2.));
  float wid=sp*(.34+.12*hl3)*(1.-.55*u);       // leaflets touch at the rachis (a full feathery blade) and taper to an open fringe at the tips (r5: a bit wider: fewer dark V gaps)
  // (r5) filtered fallback: neither a leaflet nor the gap between two leaflets gets narrower than ~2 px (sub-pixel
  // combs alias into stair-stepped dark serrations); where the gaps near the rachis would be sub-pixel the blade closes
  // (the strands repeat every .64*sp across: half period P)
  { float gmn=pg*1.0, P=sp*.32;
    float core=smoothstep(.45,.18,u)*smoothstep(gmn*1.4,gmn*.5,P-wid);
    wid=min(max(wid,gmn),P-gmn);
    wid=mix(wid,P*1.2,core); }
  float thick=.006+pg*.35;
  float slab=abs(q.y-yl)*gl-thick;
  float lfd=max(max(slab,abs(ls)-wid),lat-lw);
  float sol=smoothstep(.4,.6,pg);                     // aerial: fronds as narrow solid blades (a star from above)
  if(sol>0.) lfd=mix(lfd,max(slab-pg*.25,lat-lmax*mix(.6,1.4,smoothstep(.3,1.,pg))),sol);
  lfd=max(lfd,max(fx-Lh-.1,-fx));
  // vertical bound of the whole hanging fringe at this point of the frond: no strand reaches below ybot or above the
  // rachis. Without it the steep strands' slab distance (|dy| * small gradient factor) crawls under / over every crown.
  float ybot=Y-lmax*mix(1.42,1.4*(tV*1.45+.6*t)+1.4,sol)-.05-pg*.3;
  lfd=max(lfd,max(ybot-q.y,q.y-Y-.08-pg*.3)*gs*.85);
  tFr=vec4(t,h2,hl,u);
  return min(rach,lfd);
}
// (r9 side) palm placement from its cell, shared by palmSD and the per-cell bound pass FS_TREEB (patch_v3.cjs)
void palmGeo(vec2 c, vec4 T, out float seed, out vec3 base, out float ht, out vec2 lean){
  seed=hash12(c*1.37+.5);
  vec2 bxz=uTG.xy+(c+tJit(T))*uTG.z; base=vec3(bxz.x,T.z,bxz.y);
  float hv=hash12(c+11.3);
  ht=9.5+16.*hv;                                         // palms of every age: some tower above the broadleaf canopy, others stand in it (r5)
  lean=(hash22(c+5.1)-.5)*mix(2.,7.,hash12(c+2.7));
  vec2 ln=tLean(T); if(ln.x!=0.||ln.y!=0.){ lean=lean*.3+ln*mix(.45,.8,hash12(c+8.3)); ht=8.+10.*seed; }   // beach palms: lower, leaning out over the sand
}
float palmSD(vec3 p, vec2 c, vec4 T, out float part){
  float seed, ht; vec3 base; vec2 lean; palmGeo(c,T,seed,base,ht,lean);
  float ty=clamp((p.y-base.y)/ht,0.,1.);
  vec2 tc=base.xz+lean*(1.-(1.-ty)*(1.-ty));                  // leaves the ground slanted, grows straight up
  float slope=2.*length(lean)*(1.-ty)/ht;
  float rad=.16-.04*ty+.12*exp(-ty*ht*1.5);                    // slender, swollen foot
  float tr=(length(p.xz-tc)-rad)*inversesqrt(1.+slope*slope);
  tr=max(tr,max(base.y-.6-p.y,p.y-base.y-ht-.2))*.85;
  tr+=smoothstep(.3,.6,gPix)*.6;                               // sub-pixel trunks vanish from afar instead of aliasing into streaks
  vec3 top=vec3(base.x+lean.x,base.y+ht,base.z+lean.y);
  float fl0=mod(T.w,100.); tPalmC=vec4(top-vec3(0,fl0*.2,0),fl0);
  vec3 q=p-top;
  float fl=mod(T.w,100.);
  float hd=length(q+vec3(0,.35,0))-.42;                        // crown shaft and frond bases
  if(hd<1.2&&gPix<.15){                                         // a bunch of coconuts under the crown (near only)
    float an=atan(q.z,q.x)+seed*9.; float ia=floor(an/.8976+.5); float aa=an-ia*.8976;   // 7 sectors
    float hn=hash11(ia+seed*31.);
    vec2 pq=vec2(length(q.xz)*cos(aa)-.36-.08*hn,length(q.xz)*sin(aa));
    float nut=hn<.25?1e9:length(vec3(pq.x,q.y+.62+.22*hn,pq.y))-.13;
    hd=min(hd,nut);
  }
  float fr=length(q+vec3(0,fl*.1,0))-fl*1.2-1.5;
  part=4.;
  if(fr<1.&&gPix<.75){
    float a=atan(q.z,q.x)+seed*6.2831;
    // each whorl only where it can reach (vertical slabs around the crown): saves most of the frond cost
    float rq=length(q.xz);
    float lo0=-.85*fl-1.4, hi0=.7*fl+.3, lo1=-.95*fl-1.5, hi1=.25*fl+.3, hi2=.7*fl;
    float k0=11.+floor(4.*fract(seed*7.3)), k1=8.+floor(4.*fract(seed*11.9));   // 11-14 spreading, 8-11 drooping fronds: full coconut crowns (r5, r4)
    // 2 whorls x (nearest + neighbouring frond) + the young upright fronds, in one loop (frondSD inlined once: short compile)
    float fm=1e9; vec4 frB=vec4(0);
    for(int i=uZero;i<5;i++){
      float tier=float(min(i>>1,2)), nb=float(i&1)*(i<4?1.:0.);
      if(nb>.5&&uDbgM==53) continue;
      float lo=tier<.5?lo0:tier<1.5?lo1:-.8, hi=tier<.5?hi0:tier<1.5?hi1:hi2;
      float ob=max(lo-q.y,q.y-hi); if(tier>1.5) ob=max(ob,rq-fl*.6-.8);
      float fd=ob+.3+gPix*2.;
      if(ob<0.) fd=frondSD(q,a,tier<.5?k0:tier<1.5?k1:5.,tier<.5?0.:tier<1.5?.31:.5,seed,fl,tier,nb);
      if(fd<fm){ fm=fd; frB=tFr; }
    }
    tFr=frB;
    fr=fm*.92;
  }
  gFr=tFr;
  if(hd<fr){ fr=hd; gFr=vec4(0,.5,.5,0); }
  // aerial: the crown becomes a flat star-shaped dome (sub-pixel fronds would otherwise self-shadow into dark speckle)
  float sw=smoothstep(.4,.75,gPix);
  if(sw>0.){
    float a=atan(q.z,q.x)+seed*6.2831;
    float ka=6.+floor(4.*fract(seed*7.3));                          // (r4f: 7-8 arms as in r3: 11-14 arms are sub-pixel from the air and break into checker speckle; r5: 6-9, crowns differ)
    float x=a*ka/6.2831853, ia=floor(x+.5);
    float ha=hash11(ia+seed*17.), fa=abs(x-ia+(ha-.5)*.6);           // (r5: arms at irregular azimuths)
    float lobe=1.-smoothstep(.1,.5,fa);                              // narrow frond arms, notches between them
    // (r4f: the arms / notches fade to the mean once an arm spans fewer than ~3 px: no per-pixel checker speckle from the air)
    float armPx=4.*fl/(ka*max(gPix,.01));
    lobe=mix(.55,lobe,smoothstep(1.5,3.5,armPx));
    float rr=fl*(.55+.5*lobe*(.45+.7*ha*ha));                        // uneven frond lengths: a ragged rosette, not a starfish (r5: a few short / broken arms, shallower notches)
    float rq=clamp(length(q.xz)/max(rr,.1),0.,1.);
    // (r5) every crown tilts its own way and droops by its own amount: from the air the sunlit part of each rosette
    // differs (it was the same half-star fan on every palm)
    vec2 tv=(hash22(c+17.7)-.5)*.6;
    float dro=mix(.06,.26,hash12(c+23.1));
    float cap=sdEll(q+vec3(0,fl*(.2+dro*rq*rq)-dot(q.xz,tv),0),vec3(rr,fl*.22,rr))*.7;   // the arms droop toward their tips
    fr=sw>=1.?cap:mix(fr,cap,sw);
    if(sw>.5) gFr=vec4(.6,ha,.5,clamp(.15+.5*(1.-lobe)+.35*rq,.15,1.));   // shading: yellow arm centres, dark notches / drooping tips, a few dead arms
  }
  if(tr<fr){ part=5.; return tr; }
  return fr;
}
// inf: x=seed, y=species, z=height in crown (0..1), w=material (1 crown, 2 trunk, 4 frond, 5 palm trunk)
// bound: distance of the closest other surface (terrain): cells that cannot beat it are skipped (most of the cost)
float treesSD(vec3 p, out vec4 inf, float bound){
  vec2 ci=floor((p.xz-uTG.xy)*uTG.w);
  float best=1e9; inf=vec4(0,0,0,1);
  vec2 tsz=vec2(textureSize(uTrees,0));
  for(int k=uZero;k<9;k++){
    int kk=k==0?4:(k<5?k-1:k);   // (r9 side) the centre cell first: the nearest crowns set 'best' early, so more cells fail the bound tests
    vec2 c=ci+vec2(kk%3-1,kk/3-1);
    vec4 T=cellTex(uTrees,c);
    float R=mod(T.w,100.); if(R<.5) continue;
    vec2 bq=uTG.xy+(c+tJit(T))*uTG.z;
    if(length(p.xz-bq)-(T.w>299.?R*1.25+6.5:R*1.75+3.)>min(best,bound)) continue;   // cheap reject: cell cannot beat the best hit so far
    // (r9 side) 3D reject with the cell's baked bound (FS_TREEB): a capsule from the foot to the crown centre (palms: to the crown
    // top) that encloses trunk and crown (radius max(rad) + .75 R as in the crown test below; palms 1.6 L + 2 m). Rays skimming over
    // the canopy (view 5) no longer run treeGeo / palmSD for every crown below them. Same image (conservative bound).
    if(uDbgM!=78&&uDbgM!=80&&uDbgM!=82&&c.x>=0.&&c.y>=0.&&c.x<tsz.x&&c.y<tsz.y){ vec4 Bd=texelFetch(uTB,ivec2(c),0); if(sdCap(p,vec3(bq.x,T.z-.6,bq.y),Bd.xyz,Bd.w)>min(best,bound)) continue; }
    if(T.w>299.){ if(uDbgM==84) continue; float part; float d=palmSD(p,c,T,part); if(d<best){best=d; inf=vec4(hash12(c*1.37+.5),3.,0.,part); gFrW=gFr; gPalmC=tPalmC;} continue; }
    vec3 base,cen,rad; float sp,seed; treeGeo(c,T,base,cen,rad,sp,seed);
    float dt=sdCap(p,base-vec3(0,.5,0),cen,.14+R*.035);
    if(dt<best){best=dt; inf=vec4(seed,sp,0.,2.);}
    vec3 q=p-cen;
    if(sdEll(q,rad)-R*.75>min(best,bound)) continue;
    float ty=treeType(seed);
    float d=crownSD(p,cen,rad,seed,R,ty==3.?.26:.13+.1*fract(seed*5.3),ty==3.?.55:ty==2.?.3:.4);
    if(d<best){best=d; inf=vec4(seed,sp,clamp(q.y/rad.y*.5+.5,0.,1.),tLimb>.5?2.:1.); gClId=tClId; gClOff=tClOff; gDe=tDe; gSpr=tSpr; gLeafId=tLeafId; gLeafW=tLeafW; gInner=tInner; gCrC=vec4(cen,R);}
  }
  return best;
}
float treesSD(vec3 p, out vec4 inf){ return treesSD(p,inf,1e9); }
// shrubs and fern clumps (inf.w = 3)
float bushSD(vec3 p, out vec4 inf, float bound){
  vec2 ci=floor((p.xz-uBG.xy)*uBG.w);
  float best=1e9; inf=vec4(0,0,0,3);
  for(int j=uZero-1;j<=1;j++) for(int i=uZero-1;i<=1;i++){
    vec2 c=ci+vec2(i,j);
    vec4 T=cellTex(uBush,c);
    float R=T.w; if(R<.3) continue;
    float thk=R>9.?1.:0.; R-=10.*thk;                          // thicket shrub: raised, taller mass
    float seed=hash12(c*2.31+.7);
    vec2 bxz=uBG.xy+(c+T.xy)*uBG.z;
    vec3 cen=vec3(bxz.x,T.z+R*(.28+.5*thk),bxz.y);
    vec3 rad=vec3(R*(.85+.3*hash12(c+1.1)),R*(.55+.2*seed+.25*thk),R*(.85+.3*hash12(c+2.2)));
    vec3 q=p-cen;
    float de=sdEll(q,rad);
    if(de-R*.5>min(best,bound)) continue;
    float d=crownSD(p,cen,rad,seed*1.7+.3,R,.12+.07*fract(seed*3.7),.45);
    if(d<best){best=d; inf=vec4(seed,0.,clamp(q.y/rad.y*.5+.5,0.,1.),tLimb>.5?2.:3.); gClId=tClId; gClOff=tClOff; gDe=tDe; gSpr=tSpr; gLeafId=tLeafId; gLeafW=tLeafW; gInner=tInner; gCrC=vec4(cen,R);}
  }
  return best;
}
float bushSD(vec3 p, out vec4 inf){ return bushSD(p,inf,1e9); }
// ---- rock relief (round 5): weathered, jointed volcanic rock.
// The rock mass is cut by three joint sets: near-horizontal flow bedding and two steep joint sets, each with irregular
// spacing (every joint jittered by up to +-.32 of the spacing) and gently curved (domain warp). Every block between
// the joints stands proud or sits back by its own amount and its face is tilted by its own small angle -> planar
// facets with crisp risers, irregular block sizes, ledges that break off at the next steep joint. No cells, no
// thresholded noise (no iso-contour scribbles / topographic lines), no vertical prisms (no organ pipes).
// Four octaves (~20 m, ~8 m, ~3 m, ~1 m) in one non-unrolled loop, plus knobby weathering (smooth noise ~3 m, ~1.1 m,
// ~.45 m) and ~25 m runoff flutes on the walls. Every term is band-limited by the pixel footprint: risers widen to
// ~1.6 px, steps fade to tilted facets, and once a block / knob is < ~5 px it fades to its mean (no shimmer).
vec4 gRk=vec4(.5);  // last rockField call: x protrusion of the big block (0..1), y big-block tone, z mid-block tone, w mid-block protrusion
float gRkSl=1.;     // set by the caller: 1 on near-vertical walls (10 m slope), 0 on sloping faces
float gRkSt=1.;     // set by the caller: 1 on steep faces (> ~55 deg, 10 m slope)
float gRkC=0.;      // last rockField call: weight of the mid-block octave (footprint)
float gRkSlab=0.;   // last rockField call: weight of the big-block octave (footprint)
float gRkS10=1.;    // (r5 fix) set by shadeScene: 1 on slopes steeper than ~50 deg (10 m slope), 0 below ~27 deg
float gRkRdY=0.;    // (r5 fix) set by shadeScene: y of the view ray at the rock sample (rock albedo seen from the air)
float gRkAn=1.;     // (r5 fix) footprint stretch max(1,sqrt(.35/max(dot(N,-rd),.1))) on grazing faces, set by shadeScene for the rock normal / albedo only (1 in the march and shadows)
vec3 rkHash(vec3 c){   // integer hash (pcg3d): no near-periodicity on block indices
  uvec3 v=uvec3(ivec3(c)+ivec3(1<<20));
  v=v*1664525u+1013904223u; v.x+=v.y*v.z; v.y+=v.z*v.x; v.z+=v.x*v.y; v^=v>>16u;
  v.x+=v.y*v.z; v.y+=v.z*v.x; v.z+=v.x*v.y;
  return vec3(v)*(1./4294967296.);
}
float rkJb(float m, float sd){ return m+.64*(rkHash(vec3(m,sd,7.)).x-.5); }   // jittered joint position: spacing .36 .. 1.64
// one joint set at coordinate s (spacing units): x own block index, y own centre, z neighbour index across the nearest
// joint, w neighbour centre; returns the neighbour's blend weight (.5 on the joint, 0 beyond e)
float rkSet(float s, float sd, float e, out vec4 o){
  float m=floor(s), b0=rkJb(m,sd), b1=rkJb(m+1.,sd), k, lo, hi;
  if(s<b0){ k=m-1.; lo=rkJb(m-1.,sd); hi=b0; } else if(s>=b1){ k=m+1.; lo=b1; hi=rkJb(m+2.,sd); } else { k=m; lo=b0; hi=b1; }
  float dl=s-lo, dh=hi-s;
  if(dl<dh) o=vec4(k,(lo+hi)*.5,k-1.,(rkJb(k-1.,sd)+lo)*.5);
  else      o=vec4(k,(lo+hi)*.5,k+1.,(hi+rkJb(k+2.,sd))*.5);
  return 1.-smoothstep(-e,e,min(dl,dh));
}
// one octave of blocks. n1..n3: joint-set normals, S: their spacings (m), A: protrusion range (+-m), T: facet tilt
// (slope), r: riser half width (m, from the footprint), sd: seed.
// returns x displacement (m, mean 0), y protrusion 0..1, z tone 0..1
vec3 rkBlocks(vec3 q, vec3 n1, vec3 n2, vec3 n3, vec3 S, float A, float T, float r, float sd, float em){
  vec3 s=vec3(dot(q,n1),dot(q,n2),dot(q,n3))/S;
  vec3 e=clamp(r/S,em,max(.2,em));   // (r5 fix) em: minimum riser half width (spacing units), .07 except for the buttress blocks   // r = .8 x footprint (m): risers ~1.6 px wide, never steeper than the march can follow
  vec4 o2, o3;
  float w2=rkSet(s.y,sd+11.,e.y,o2), w3=rkSet(s.z,sd+23.,e.z,o3);
  vec3 acc=vec3(0);
  for(int m=uZero;m<4;m++){            // the (up to 4) columns around q, bounded by the two steep joint sets
    float bj=float(m&1), bk=float(m>>1);
    float wc=mix(1.-w2,w2,bj)*mix(1.-w3,w3,bk);
    if(wc<.003) continue;
    vec2 kc=vec2(mix(o2.x,o2.z,bj),mix(o3.x,o3.z,bk)), cc=vec2(mix(o2.y,o2.w,bj),mix(o3.y,o3.w,bk));
    vec3 hc=rkHash(vec3(kc,sd*53.+17.));
    // every column has its own bedding offset and dip: the flat joints end at the steep ones (T-junctions),
    // no joint runs across the whole face as a ruled line
    float s1=s.x+hc.z*4.37+(fract(hc.z*9.71)-.5)*.34*(s.y-cc.x)*S.y/S.x+(hc.y-.5)*.34*(s.z-cc.y)*S.z/S.x;
    vec4 o1; float w1=rkSet(s1,sd,e.x,o1);
    for(int l=uZero;l<2;l++){
      float bl=float(l), w=wc*mix(1.-w1,w1,bl);
      if(w<.003) continue;
      vec3 hh=rkHash(vec3(mix(o1.x,o1.z,bl),kc)+vec3(sd*131.,sd*71.,sd*37.));
      vec3 tl=rkHash(hh*4096.+vec3(5.,9.,sd))*2.-1.;
      // blocks of one column share part of their protrusion: steps across the flat joints are smaller than across
      // the steep ones (buttresses and gullies, broken ledges; not a stepped pyramid of shelves)
      float dsp=A*(mix(hc.x,hh.x,.5)*2.-1.)+T*dot(tl,vec3((s1-mix(o1.y,o1.w,bl))*S.x,(s.y-cc.x)*S.y,(s.z-cc.y)*S.z));
      acc+=w*vec3(dsp,mix(hc.x,hh.x,.5),hh.y);
    }
  }
  return acc;
}
float rockField(vec3 p){
  // gently curved joints: warp the rock mass (~50 m)
  vec3 q=p+vec3(noise3(p*.021+1.3),noise3(p*.017+4.2)*.6,noise3(p*.021+7.1))*6.;
  float gp=max(gPix*gRkAn,1e-4);   // (r5 fix) anisotropic footprint on grazing faces (shading only): no fine horizontal hatching
  // footprint band limit per octave (k = r / smallest spacing, r = .8 px in m): the steps between blocks (A) only
  // while the blocks are well resolved (from afar a mesh of thin joint lines reads as a quilt / tile pattern); the
  // tilted facets alone down to ~6 px; then the octave fades to its mean (gone at ~4 px)
  float r=.8*gp;
  // The four octaves run in ONE non-unrolled loop (uZero): rkBlocks is instanced once per rockField call site instead
  // of four times (rockField is inlined in every sceneSD caller: the driver shader compile was taking minutes).
  //  o=0 buttress-scale blocks (~14-24 m): big facets and recesses that shape the faces seen from the air
  //  o=1 big blocks (~6-10 m, taller than wide): flow beds dipping a few degrees, two steep joint sets ~70 deg apart;
  //      steps only while a block spans > ~20 px (k .04-.08), tilted facets below (no joint-line mesh from the air)
  //  o=2 mid blocks (~2-3.5 m), rotated joint sets (steps kept down to ~5 px: the broken texture of r2's pillars)
  //  o=3 small blocks (~0.7-1.1 m): close views only
  vec3 B0=vec3(0,.5,.5), B1=B0, B2=B0; float B3=0., w1=0., w2=0.;
  for(int o=uZero;o<4;o++){
    vec3 n1, n2, n3, S, off; float A, T, sd, f0, f1;
    if(o==0){      n1=vec3(-.05,.99,.1);  n2=vec3(.8,.08,.6);    n3=vec3(-.7,.1,.7);    S=vec3(24.,14.,18.); off=vec3(11.3); A=1.6; T=.15; sd=4.; f0=.045; f1=.09; }   // (r5 fix) A 1.8, T .2, steps faded .06-.11: big planar facets / quilt panels from the air
    else if(o==1){ n1=vec3(.10,.99,.07);  n2=vec3(.95,.12,.30);  n3=vec3(-.40,.07,.91); S=vec3(10.,6.5,8.5); off=vec3(0.);   A=1.4; T=.16; sd=1.; f0=.04; f1=.08; }
    else if(o==2){ n1=vec3(-.08,.99,.12); n2=vec3(.60,.10,-.79); n3=vec3(.86,.05,.51);  S=vec3(3.4,2.2,2.8); off=vec3(3.7);  A=.42; T=.22; sd=2.; f0=.08; f1=.13; }
    else {         n1=vec3(.15,.98,-.10); n2=vec3(.30,.12,.95);  n3=vec3(.97,.0,-.25);  S=vec3(1.1,.7,.85); off=vec3(1.9);  A=.15; T=.26; sd=3.; f0=.06; f1=.11; }
    float k=r/S.y, w=1.-smoothstep(.13,.2,k);   // S.y = the smallest spacing of the octave
    if(w<=0.) break;                             // the octaves get smaller: all further ones are faded too
    // (r5 it2) from o=1 on the joints wander (~7 m, +-.8 m): broken, irregular joint traces instead of ruled lines
    if(o==1) q+=vec3(noise3(q*.14+2.),noise3(q*.14+5.)*.7,noise3(q*.14+8.))*.8*w;
    // (r5 fix) from the air the big / buttress blocks read as a quilt of planar panels with crisp edges: wider risers
    // (~2.5 px at gPix >~1 m) and the B1 facet tilt fades once its blocks span < ~12 px (the noise relief below carries it)
    float rm=o<2?1.+.7*smoothstep(.45,1.1,gp):1.;
    vec3 b=rkBlocks(q+off,n1,n2,n3,S,A*(1.-smoothstep(f0,f1,k)),T*(o==1?1.-.6*smoothstep(.06,.12,k):1.),r*rm,sd,o==0?mix(.09,.14,smoothstep(.3,.9,gp)):.07);
    if(o==0) B0=mix(B0,b,w); else if(o==1){ B1=mix(B1,b,w); w1=w; } else if(o==2){ B2=mix(B2,b,w); w2=w; } else B3=b.x*w;
  }
  // big runoff flutes (~25 m) on the near-vertical walls: rounded ribs between gullies running down the face
  float fn=noise3(vec3(q.x*.04,q.y*.006,q.z*.04));
  float fl=1.-sqrt(fn*fn+.03); fl*=fl;
  // knobby, pocked weathering of the block faces (r3): two octaves of smooth noise (~1.1 m, ~.45 m), each faded once a
  // knob spans fewer than ~10 px (gone at 5 px: no shimmer); plain noise, never thresholded (no iso-line scribbles)
  // (it2) plus a ~3 m octave: the block faces are never flat planes at mid range (r3 right cliff)
  float fc=clamp(3./(gp*5.)-1.,0.,1.), fa=clamp(1.1/(gp*5.)-1.,0.,1.), fb=clamp(.45/(gp*5.)-1.,0.,1.);
  float ru=0.;
  if(fc>0.) ru=.26*noise3(q*.33+5.)*fc;
  // (r5 fix) the ~20 m facets were flat planes at 100-400 m (v3 right cliff) and panels from the air: ~7 m bulges and
  // hollows (faded below ~5 px), plus on the walls ~6 m vertical ribs / runnels (r3: fluted, columnar cliff faces;
  // ridged noise stretched 7x along y, faded below ~5 px)
  float gq=max(gPix*min(gRkAn,1.5),1e-4);   // smooth noise: hardly aliases, keep it on grazing faces
  float f7=clamp(7./(gq*3.5)-1.,0.,1.), f6=clamp(6./(gq*5.)-1.,0.,1.)*gRkSl, f35=clamp(3.5/(gq*3.5)-1.,0.,1.);
  if(f7>0.) ru+=.8*noise3(q*.14+11.)*f7;
  if(f35>0.) ru+=.45*(.35-abs(noise3(q*.29+23.)))*f35;   // ~3.5 m ridged crags / pockets: rough, not clay, from the air
  if(f6>0.) ru+=1.3*(.35-abs(noise3(vec3(q.x*.17,q.y*.024,q.z*.17)+17.)))*f6;
  if(fa>0.) ru+=.2*noise3(q*.9+3.)*fa+(fb>0.?.08*noise3(q*2.2+8.)*fb:0.);
  gRk=vec4(mix(B0.y,B1.y,w1),mix(B0.z,B1.z,w1*.7),B2.z,B2.y); gRkC=w2; gRkSlab=w1;
  return .6+B0.x+B1.x+B2.x+B3+fl*1.5*gRkSl+ru;
}
float rockDisp(vec3 p){ return rockField(p); }
// Open joints between the columns of a wall (r2: dark slots between the lit pillars), shading only. Two steep joint
// sets (~3.6 m / ~4.6 m, every joint jittered), each joint open over parts of its length only and at its own width
// (.1-.45 m); returns the slot coverage 0..1. Band-limited: once a slot is narrower than ~1 px it fades to its mean
// darkening (no crawling lines in motion).
float rkSlotD(float s, float sd, out float id){   // distance (spacing units) to the nearest jittered joint, its index
  float m=floor(s), best=9.; id=m;
  for(int k=uZero-1;k<=2;k++){ float b=rkJb(m+float(k),sd), d=abs(s-b); if(d<best){ best=d; id=m+float(k); } }
  return best;
}
float rkSlots(vec3 q, float gp){
  vec3 w=q+vec3(noise3(q*.23+2.),0.,noise3(q*.23+6.))*.7;   // joints wander a little
  float cav=0.;
  for(int j=uZero;j<2;j++){
    vec3 n=j==0?vec3(.93,0.,.37):vec3(-.47,0.,.88);
    float S=j==0?3.6:4.6, sd=j==0?61.:73.;
    float id, d=rkSlotD(dot(w,n)/S,sd,id)*S;
    vec3 h=rkHash(vec3(id,sd,3.));
    float wd=mix(.1,.45,h.x*h.x);
    // open over runs of 4-15 m along the joint (mostly vertical), closed elsewhere
    float op=smoothstep(-.05,.3,noise3(vec3(w.y*.08,dot(w,vec3(-n.z,0.,n.x))*.15,id*1.7+sd))+(h.y-.5)*.7);
    float vis=clamp(wd/gp*1.2-.3,0.,1.);
    float sl=(1.-smoothstep(wd*.35,wd+gp*.6,d))*vis+(1.-vis)*min(1.6*wd/S,1.);
    cav=max(cav,sl*op);
  }
  return cav;
}
// slope / understory canopy on jittered 3D cells (works on vertical walls).
// Shrub / fern carpet: small parabolic domes (2x2x2 search).
// x = canopy height here (m), y = crown id, z = this crown's own top height (m)
vec3 canopyCrowns(vec3 p, float cs, float seed, float hs, float pres){
  vec3 g=p/cs+seed, n=floor(g), f=fract(g);
  float best=0., id=0., top=1.;
  float lob=noise3(g*2.3+1.7)*.4;                             // (r5) lobed, ragged clump outlines (shared field), not round pebbles
  for(int k=uZero;k<8;k++){
    vec3 o=vec3(k&1,(k>>1)&1,(k>>2)&1)+step(.5,f)-1.;
    vec3 c=n+o;
    vec3 h=hash33(c*1.13+7.+seed);
    if(fract(h.z*13.7)>pres) continue;
    vec3 r=o+.2+.6*h-f;
    float rad=.26+.5*h.x*h.x;                                  // (r5: many small clumps, a few big ones; was .34-.74 evenly: identical pebbles)
    float A=rad*hs*(.5+.5*fract(h.y*7.31));
    float hh=A*(1.-dot(r,r)/(rad*rad)*(1.+lob));
    if(hh>best){ best=hh; id=h.y; top=A; }
  }
  return vec3(best*cs,id,top*cs);
}
// Forest: packed crowns of very different sizes. Large overlapping domes (3x3x3 search), each truncated
// by its neighbours, so the canopy is a closed mosaic of crowns with dark creases, not balls on a lawn.
vec3 forestCrowns(vec3 p, float cs){
  vec3 g=p/cs, n=floor(g), f=fract(g);
  float best=-1e9, id=0., top=1.;
  float lob=noise3(p*(1.6/cs)+4.1)*.6+noise3(p*(3.3/cs)+1.3)*.3;   // ragged crown outlines (shared field)
  for(int k=uZero;k<27;k++){
    vec3 o=vec3(k%3,(k/3)%3,k/9)-1.;
    vec3 c=n+o;
    vec3 h=hash33(c*1.31+2.7);
    vec3 r=o+.1+.8*h-f;
    float big=h.x*h.x;
    float rad=.6+.6*big;                        // a few big emergent crowns among many small ones
    float A=rad*(.2+.15*fract(h.y*5.37))*(1.+.5*big);        // understory under the instanced trees: lower than their crowns
    float hh=A*(1.-dot(r,r)/(rad*rad)*(1.+.55*lob));
    if(hh>best){ best=hh; id=h.y; top=A; }
  }
  return vec3(max(best,0.)*cs,id,top*cs);
}
float gCanW, gCanAO;   // forest-layer weight (or 1 on a scattered tree) and leaf-mass crevice occlusion of the last canopyField call
// canopy fields are evaluated at the point projected onto the terrain surface (along the terrain normal),
// so every crown is a pure height offset: no floating blobs / overhangs on steep walls
// (smooth normal from 3 m central differences: the bilinear height field's kinks would otherwise streak the crowns)
vec3 smoothNormal(vec2 p){
  const float e=3.;
  return normalize(vec3(H(p-vec2(e,0.))-H(p+vec2(e,0.)),2.*e,H(p-vec2(0.,e))-H(p+vec2(0.,e))));
}
vec3 canopyAnchor(vec3 p, float h, vec3 n){ return p-n*((p.y-h)*n.y); }
vec3 canopyAnchor(vec3 p, float h){ return canopyAnchor(p,h,smoothNormal(p.xz)); }
vec3 canopyField(vec3 p, vec3 pa, float fw){   // pa = p anchored on the terrain surface (for the carpet on walls)
  vec3 r=vec3(0,0,1);
  float w=smoothstep(.1,.9,fw);
  gCanAO=1.;
  if(w>.01){
    r=forestCrowns(p,6.5);   // (r8 forest) 9 m cells -> 6.5: from the air the slope canopy is a fine crown texture, not 15-20 m broccoli domes
    // each crown is a head of leaf masses (~2.6 m) with dark crevices between them
    float fm=clamp(3.2/max(gPix,1e-4)*.3-1.,0.,1.)*smoothstep(.0,.2,r.x/r.z);   // leaf masses fade out below ~7 px (no bubble texture from afar)
    if(fm>0.){ vec3 pm=p+1.3; if(gPix<.66) pm+=noised3(p*.55+7.7).yzw*.45*clamp(2.-gPix*3.,0.,1.);   // (r5) warped, varied leaf masses: ragged lobes, not a bed of round 3 m pebbles
      float s=sprayDomeV(pm,3.2); r.x+=(s-.43)*1.4*fm; gCanAO=mix(1.,(.6+.5*s)/.84,fm*w); }   // (r5: crevice tone .55..1.6 -> .71..1.31: no dark outline round every mass)
  }
  if(w<.99){
    float sa=clamp(2.4/gPix/5.-1.,0.,1.);    // shrub domes fade out below ~10 px: the fern / grass carpet is smooth from afar
    vec3 cS=canopyCrowns(pa+vec3(noise3(pa*.35+4.),0.,noise3(pa*.35+9.))*1.1,2.4,17.3,.38*(1.-.45*uMC.x),.85);   // warped, gappy: irregular fern / shrub clumps, not fish scales (r5: flatter, .5 -> .38: no cobbles)
    float cl=(noise3(pa*.13+1.7)*.5+.5)*1.2;   // soft 4-8 m clumps of the shrub / fern carpet
    cS.x=cS.x*sa+cl; cS.z=cS.z*sa+1.2;
    // scattered small trees standing out of the fern / shrub carpet (darker crowns, r9 / r10)
    // groves (irregular clumps, open fern between them); lumpy crowns, tall enough to shade the fern around them
    vec3 pw=pa+vec3(noise3(pa*.2+1.),0.,noise3(pa*.2+6.))*2.6;   // warped: irregular crown outlines, not discs
    if(gPix<1.2){ vec3 wg=noised3(pa*.75+3.3).yzw; pw.xz+=wg.xz*.4*clamp(2.-gPix*1.6,0.,1.); }   // (r5) + a finer warp (~1.3 m): ragged, lobed crown edges with protruding clumps, not round discs (faded once sub-pixel)
    vec3 cT=canopyCrowns(pw,5.2,41.3,1.05,mix(.55+.35*uVD.z,.97+.03*uVD.z,smoothstep(-.25,.35,noise3(pa*.022+8.)+.3*noise3(pa*.07+3.))));   // (r8 forest) .22-.92 -> .55-.97: trees dominate the walls, the lime fern is the minority between them (r9, r5: dark crowns everywhere)   // trees scattered all over the walls, denser in groves (r9: distinct crowns everywhere)
    if(cT.x>0.){ float lm=clamp(2.2/max(gPix,1e-4)*.3-1.,0.,1.)*smoothstep(.0,.3,cT.x/cT.z); cT.x=max(cT.x+(sprayDomeV(pa+2.7,2.2)-.45)*1.4*lm,0.); }
    float tw=0.;
    if(cT.x>cS.x-.6){ tw=smoothstep(-.3,.6,cT.x-cS.x); float hk=max(.6-abs(cT.x-cS.x),0.); cS=vec3(max(cS.x,cT.x)+hk*hk*.4,cT.y,max(cS.z,cT.z)); }   // smooth union: no crease ring at the crown foot
    r=vec3(mix(cS.x,r.x,w),w>.5?r.y:cS.y,mix(cS.z,r.z,w));
    w=max(w,tw*(1.-w));
  }
  gCanW=w;
  return r;
}
// (r13 mat) uMC.y (URL mc1): the vegetation mask (2.3 m texels) seen up a near-vertical wall: every texel edge ran down the face as a
//  stair-stepped cover boundary (stepped black slots of bare rock in the film, t72 / t84). On steep ground (1.2 m slope > ~45-70 deg)
//  the mask is read at a point displaced by a 3D noise (+-2.6 m, ~3 m features that change with height), so the boundary is ragged up
//  the face. Material only (shadeScene): in terrSD two noise3 per call cost +45 % render time (v1 982 -> 1426 ms, v5 1865 -> 2839 ms),
//  and a cheap crossed-sine displacement drew regular stripes over the vegetated walls (v5 / v9). The canopy relief keeps the
//  undisplaced mask; the cover / rock colour boundary is ragged.
vec2 vegWarp(vec3 p, float s){ float w=uMC.y*smoothstep(1.,2.6,s); return w>0.?vec2(noise3(p*.33+1.7),noise3(p*.33+8.3))*2.2*w:vec2(0.); }
float gDTrue=0.;   // true distance to the surface at the last terrSD / sceneSD evaluation (the returned field is scaled down for safe steps)
float terrSD(vec3 p, float h, out float steepW){
  const float e=1.2;
  gCarpIn=0.;   // (r9 side) never stale (read back at the primary hit through gCarpHit)
  float hx=H(p.xz+vec2(e,0.))-h, hz=H(p.xz+vec2(0.,e))-h;
  float s2=(hx*hx+hz*hz)/(e*e);
  float ms=textureLod(uMS,(p.xz-uRect.xy)*uRect.w,0.).r;      // max slope nearby: keeps steps conservative next to cliffs
  float d=(p.y-h)/sqrt(1.+max(s2,ms*ms));
  steepW=smoothstep(.8,1.6,sqrt(s2));
  if(d>=12.+1.5*gEpsM){ steepW=0.; gDTrue=d-10.; return d-10.; }   // conservative bound for the relief below (r5 fix: the value returned at the threshold, 2 m + 1.5 gEpsM, stays above the march hit tolerance gEpsM as in crownSD / leafOct -- a plain 2 m was hit as a false shell ~10 m above far ridges on grazing rays once gEpsM > 2 m)
  {
    vec4 vm=VEG(p.xz);
    float bare=1.-smoothstep(.25,.75,vm.r);
    // rock relief weight from a wide-baseline slope (10 m) and the sample height: the relief shell reaches ~5 m in
    // front of a wall, where the local 1.2 m slope and the ground height h are those of the foot of the wall;
    // weighting by them switched metres of relief on and off between neighbouring samples (speckled normals / AO)
    const float e3=5.;
    float sx=H(p.xz+vec2(e3,0.))-H(p.xz-vec2(e3,0.)), sz=H(p.xz+vec2(0.,e3))-H(p.xz-vec2(0.,e3));
    float sl10=sqrt(sx*sx+sz*sz)/(2.*e3);
    float steepB=smoothstep(.6,1.3,sl10);
    gRkSl=smoothstep(1.2,2.2,sl10); gRkSt=smoothstep(1.,2.2,sl10);
    float rw=steepB*bare*smoothstep(-.5,2.5,max(h,p.y))*smoothstep(-1.5,1.5,p.y);   // (r5: relief down to the waterline: the rock there is the most rugged)
    // rock: the base surface is the C2 bicubic height, not the bilinear one. On a wall the bilinear cells' normal kinks
    // read as evenly spaced horizontal / vertical light lines (~3-4 m apart: the 1.17 m grid seen up a steep face)
    // The relief offset is scaled with the slope of that smooth surface too: at a sheer cliff foot the bilinear 1.2 m
    // slope jumps from cell to cell, so the wall stood out by a different amount in every cell column ("organ pipes")
    float hw=steepB*mix(bare,1.,uMC.z), s2r=s2;   // (r13 mat) uMC.z (URL mc2): the vegetated walls too sit on the C2 bicubic surface (the bilinear 1.2 m cells read as horizontal stepped slots up the walls, film t72 / t84)
    if(hw>.01){
      float hc0=Hc(p.xz), hcx=Hc(p.xz+vec2(e,0.))-hc0, hcz=Hc(p.xz+vec2(0.,e))-hc0;
      s2r=mix(s2,(hcx*hcx+hcz*hcz)/(e*e),hw);
      d-=(hc0-h)*hw/sqrt(1.+max(s2,ms*ms));
    }
    if(rw>.01) d=(d-rockDisp(p)*rw*.9*sqrt((1.+s2r)/(1.+max(s2,ms*ms))))*mix(1.,.6,rw);   // true perpendicular offset; steep relief -> shorter steps (no overshoot)
    gDTrue=d/mix(1.,.6,rw)*sqrt((1.+max(s2,ms*ms))/(1.+s2r));
    float cw=smoothstep(.2,.6,vm.r)*smoothstep(.35,.0,vm.b)*(gVeg&&uDbgM!=85?1.:0.);   // (r9 side) 84 / 85: timing switches (palms / ground canopy off)
    if(cw>.01){
      vec3 sn=smoothNormal(p.xz);
      vec3 cc=canopyField(p,canopyAnchor(p,h,sn),vm.a);
      // the canopy offset is applied in true-distance units (smooth local slope); the conservative max-slope
      // factor (sc) only shortens the march steps. Otherwise crowns near cliff edges inflate into floating slabs.
      float kt=sqrt(1.+max(s2,ms*ms))*sn.y;                    // scaled -> true distance
      d*=kt;
      float sc=min(1./max(kt,1e-3),1.);
      float dc=d-(cc.x+.3)*cw;                                 // canopy surface
      float dm=dc-foliageRelief(p,1.)*cw*mix(clamp(2.2/gPix*.25-1.,0.,1.),1.,gCanW);   // fern / grass carpet: smooth from afar; tree crowns keep their leaf masses
      float lw=clamp(3.-gPix*20.,0.,1.)*cw;                   // leaves out to gPix ~.15 (the shadow pass, at gPix .32, stays cheap)
      if(uDbgM==20) lw=0.;
      if(lw>0.){ float dl=leafOct(p,.17,5.,dm+.05,2.); gCarpIn=max(smoothstep(-.03,.03,dl-dm-.6),smoothstep(.0,.5,-dm))*lw; dm=mix(dm,min(dl,dm+.6),lw); }
      d=dm*sc*(gPix<.3?.55:.72); gDTrue=dm;                               // coarse LOD (incl. the shadow pass): smoother field, longer steps
    }
    steepW=rw;
  }
  return d;
}
// whole scene: distance + info (w = material: 0 ground/rock, 1 crown, 2 trunk, 3 bush, 4 frond, 5 palm trunk)
float sceneSD(vec3 p, out vec4 inf){
  float h=H(p.xz);
  float sw;
  float d=terrSD(p,h,sw);
  inf=vec4(sw,0.,0.,0.);
  if(gVeg&&p.y-h<ENV){
    float dT0=gDTrue;
    vec4 ti; float dt=uDbgM==22?1e9:treesSD(p,ti,d); if(dt<d){d=dt; inf=ti; dT0=d;}
    vec4 bi; float db=uDbgM==23?1e9:bushSD(p,bi,d); if(db<d){d=db; inf=bi; dT0=d;}
    gDTrue=dT0;
  }
  return d;
}

// ---------------------------------------------------------------- terrain march / shadows
float gWhy=0., gIt=0.;
// Two-level march: cheap heightfield steps far above the vegetation envelope, sphere tracing inside it.
// Steps grow with distance and iteration count so grazing rays along steep slopes always terminate.
float pkDbg(){ return uDbgM>=100?float(uDbgM-100):1.; }   // debug only (dbgm=100+k): LOD of a k x wider view, to inspect far LOD in a zoomed probe
bool march(vec3 ro, vec3 rd, float tmin, float tmax, out float tHit, out vec4 inf){
  float t=tmin;
  // step length scaled by a random factor per pixel, sample and frame: whether a step skips or lands in a thin canopy / relief
  // layer on a steep slope was decided by the camera-relative step sequence, the same for whole bands of pixels, so bands of
  // shadow crawled down the ridges as the camera moved (~0:35); random per ray, the accumulated samples average it instead
  float rj=mix(.7,1.2,hash12(gl_FragCoord.xy*1.37+vec2(fract(uTime*.1)*997.,fract(uTime*.37)*613.)));
  float pk=uTan*2./uRes.y*pkDbg();
  inf=vec4(0);
  float lastD=1e9;
  bool inEnv=false;
  for(int i=uZero;i<1100;i++){
    gIt=float(i);
    if(t>tmax){ gWhy=1.; gEpsM=0.; return false; }
    vec3 p=ro+rd*t;
    float hgt=p.y-H(p.xz);
    if(hgt>ENV){ t+=max((hgt-ENV)*.5*rj,.3+t*.0015); inEnv=false; continue; }
    inEnv=true;
    gPix=t*pk; gVeg=t<uVegRange;
    gEpsM=gPix*(.6+float(i)*.004);
    float d=sceneSD(p,inf);
    float eps=gPix*(.6+float(i)*(inf.w>3.5&&inf.w<4.5?.0015:.004));   // palm fronds: no growing hit tolerance (it fattened the see-through fringes into curtains)
    if(d<eps){   // terrain hits: Newton steps on the true distance to a fixed two pixel footprints off the surface. The scaled-down field
      // stopped short by a view-dependent ~0.5-6 footprints (eps grows with the step count, d shrinks with slope), so on steep
      // canopy the shaded point floated in front of the slope and slid across it as the camera moved: crawling, banded shadows
      float ca=max(dot(smoothNormal(p.xz),-rd),.25), off=gPix*2.; bool above=p.y-hgt>1.;   // not under the sea: a lifted hit left the water test
      for(int k=uZero;k<3;k++){ float g=gDTrue; t+=above&&inf.w<.5&&abs(g)<1e4?clamp((g-off)/ca,-2.,6.):d*.9; d=sceneSD(ro+rd*t,inf); }
      tHit=t; gEpsM=0.; gCarpHit=gCarpIn; return true; }
    t+=max(d*.8*rj,eps*.5);
    lastD=d;
  }
  gWhy=3.; gEpsM=0.;
  if(inEnv){ tHit=t; gWhy=4.; gCarpHit=gCarpIn; return true; }   // budget exhausted inside the vegetation layer: accept (never leave a hole)
  return false;
}
float shadowH(vec3 p, vec3 L){
  float res=1., t=.5;
  for(int i=uZero;i<70;i++){
    vec3 q=p+L*t;
    if(q.y>uHmax) break;
    float d=q.y-H(q.xz);
    res=min(res,10.*d/t);   // (r7 forest) crisper ridge / valley-wall shadow edges (6: very soft, smeared the lit / shaded faces)
    if(res<-.2) break;
    t+=clamp(d*.5,.8,40.);
  }
  return smoothstep(-.2,1.,res);
}
// geometry shadows near (trees, bushes, rock relief), heightfield beyond
float gShStart=.06;
float gShG=1.;   // (r7 forest) the near-geometry part (crowns, bushes, rock relief) of the last shadow() call
float shadow(vec3 p, vec3 L, float range){
  float res=1., t=gShStart;
  gShG=1.;
  if(range>0.){
    float gp=gPix; gPix=max(gPix,.32);   // shadows use the cheap crown LOD
    float kP=mix(13.,10.,smoothstep(.6,1.6,gp));   // (r7 forest) crisp crown shadows close up, a little softer where the crowns are only a few px (no crawling pixel shadows)
    for(int i=uZero;i<48;i++){
      vec3 q=p+L*t;
      vec4 inf; float d=sceneSD(q,inf);
      res=min(res,kP*d/t);   // (r7 forest) crisper crown shadows (9: soft grey penumbrae flattened the canopy)
      if(res<.01){ gPix=gp; gShG=0.; return 0.; }
      t+=clamp(d,.08,3.5);
      if(t>range) break;
    }
    gPix=gp;
  }
  gShG=clamp(res,0.,1.);
  return min(res,shadowH(p+L*t,L));
}

