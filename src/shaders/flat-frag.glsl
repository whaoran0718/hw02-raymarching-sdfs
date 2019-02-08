#version 300 es
precision highp float;

uniform vec3 u_Eye, u_Ref, u_Up;
uniform vec2 u_Dimensions;
uniform float u_Time;
uniform vec4 u_Color;
uniform int u_Petal;

in vec2 fs_Pos;
out vec4 out_Col;

float maxMarchLength = 1e3;
float epsilon = 1e-2;
int maxMarchIter = 128;
vec3 lightVec = vec3(-1, -1, -1);
float bloom = 0.0;

vec2 c_seed = vec2(0);
float PI = 3.14159265;
float PI_2 = 6.2831853;

float random1( vec2 p , vec2 seed) {
  return fract(sin(dot(p + seed, vec2(127.1, 311.7))) * 43758.5453);
}

vec2 random2( vec2 p , vec2 seed) {
  return fract(sin(vec2(dot(p + seed, vec2(311.7, 127.1)), dot(p + seed, vec2(269.5, 183.3)))) * 85734.3545);
}

// Perlin Noise
/////////////////////////////////////////
// Falloff founction from CIS566 course slides
float falloff(float t) {
  t = t * t * t * (t * (t * 6. - 15.) + 10.);
  return t;
}

vec2 randGrad(vec2 p, vec2 seed) {
  float randDeg = random1(p, seed) * PI_2;
  return vec2(cos(randDeg), sin(randDeg));
}

float PerlinNoise(vec2 p, float s) {
    p /= s;
    vec2 pCell = floor(p);
    p -= pCell;
    float dotGrad00 = dot(randGrad(pCell + vec2(0., 0.), c_seed), p - vec2(0., 0.));
    float dotGrad01 = dot(randGrad(pCell + vec2(0., 1.), c_seed), p - vec2(0., 1.));
    float dotGrad10 = dot(randGrad(pCell + vec2(1., 0.), c_seed), p - vec2(1., 0.));
    float dotGrad11 = dot(randGrad(pCell + vec2(1., 1.), c_seed), p - vec2(1., 1.));

    return mix(mix(dotGrad00, dotGrad10, falloff(p.x)), mix(dotGrad01, dotGrad11, falloff(p.x)), falloff(p.y)) * .5 + .5;
}


// FBM Noise
/////////////////////////////////////////
float FBMPerlin(vec2 p, float minCell, int maxIter) {
    float sum = 0.;
    float noise = 0.;
    for (int i = 0; i < maxIter; i++) {
        noise += PerlinNoise(p, minCell * pow(2., float(i))) / pow(2., float(maxIter - i));
        sum += 1. / pow(2., float(maxIter - i));
    }
    noise /= sum;
    return noise;
}


// Rotation Matrix Inverse
////////////////////////////////////////////////////////////////////////////////////////////
mat3 rotMatInv(vec3 angle) {
  vec3 rad = radians(angle);

  float cX = cos(rad.x);
  float sX = sin(rad.x);
  mat3 rX = mat3(1.0, 0.0, 0.0,
                 0.0, cX, -sX,
                 0.0, sX, cX);

  float cY = cos(rad.y);
  float sY = sin(rad.y);
  mat3 rY = mat3(cY, 0.0, sY,
                 0.0, 1.0, 0.0,
                 -sY, 0.0, cY);

  float cZ = cos(rad.z);
  float sZ = sin(rad.z);
  mat3 rZ = mat3(cZ, -sZ, 0.0,
                 sZ, cZ, 0.0,
                 0.0, 0.0, 1.0);
  
  return rZ * rX * rY;
}

// SDF
// Reference from http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
// & http://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
////////////////////////////////////////////////////////////////////////////////////////////
float sdBox( vec3 p, vec3 b )
{
  vec3 d = abs(p) - b;
  return length(max(d,0.0)) + min(max(d.x,max(d.y,d.z)),0.0);
}

vec2 sdVesica(vec2 p, float r, float d, float label)
{
  vec2 po = p;
  p = abs(p);

  float b = sqrt(r*r-d*d);
  float dist = ((p.y-b)*d > p.x*b) ? length(p-vec2(0.0,b)) : length(p-vec2(-d,0.0))-r;
  float noise = smoothstep(0.0, 0.9, FBMPerlin(po, 0.2, 4)) * 0.8 + 0.2;
  label += smoothstep(0.1, 0.6, clamp(po.y / b * 0.5 + 0.5, 0.0, 1.0) * (1.0 - clamp(0.0, 1.0, abs(dist) / b))) * noise;
  return vec2(dist, label);
}

vec3 opCheapBendX(vec3 p, float w)
{
  float c = cos(w*p.y);
  float s = sin(w*p.y);
  mat2  m = mat2(c, s, -s, c);
  return vec3(p.x, m * p.yz);
}

vec3 opCheapBendY(vec3 p, float w)
{
  float c = cos(w*p.x);
  float s = sin(w*p.x);
  mat2  m = mat2(c, -s, s, c);
  vec2 xz = m * p.xz;
  return vec3(xz.x, p.y, xz.y);
}

vec3 opCheapBendZ(vec3 p, float w)
{
  float c = cos(w*p.y);
  float s = sin(w*p.y);
  mat2  m = mat2(c, s, -s, c);
  return vec3(m*p.xy, p.z);
}

float opExtrussion(vec3 p, float sdf, float h )
{
  vec2 w = vec2( sdf, abs(p.z) - h );
  return min(max(w.x,w.y),0.0) + length(max(w,0.0));
}

float opRound(float d, float rad)
{
    return d - rad;
}

float sdRoundCone( vec3 p, float r1, float r2, float h )
{
    vec2 q = vec2( length(p.xz), p.y );
    
    float b = (r1-r2)/h;
    float a = sqrt(1.0-b*b);
    float k = dot(q,vec2(-b,a));
    
    if( k < 0.0 ) return length(q) - r1;
    if( k > a*h ) return length(q-vec2(0.0,h)) - r2;
        
    return dot(q, vec2(a,b) ) - r1;
}

float sdSphere( vec3 p, float s )
{
  return length(p)-s;
}

float sdBezier( in vec2 pos, in vec2 A, in vec2 B, in vec2 C )
{   
  vec2 a = B - A;
  vec2 b = A - 2.0*B + C;
  vec2 c = a * 2.0;
  vec2 d = A - pos;
  float kk = 1.0 / dot(b,b);
  float kx = kk * dot(a,b);
  float ky = kk * (2.0*dot(a,a)+dot(d,b)) / 3.0;
  float kz = kk * dot(d,a);      
  float res = 0.0;
  float p = ky - kx*kx;
  float p3 = p*p*p;
  float q = kx*(2.0*kx*kx - 3.0*ky) + kz;
  float h = q*q + 4.0*p3;
  if(h >= 0.0) 
  { 
      h = sqrt(h);
      vec2 x = (vec2(h, -h) - q) / 2.0;
      vec2 uv = sign(x)*pow(abs(x), vec2(1.0/3.0));
      float t = uv.x + uv.y - kx;
      t = clamp( t, 0.0, 1.0 );
      vec2 qos = d + (c + b*t)*t;
      res = dot(qos,qos);
  }
  else
  {
      float z = sqrt(-p);
      float v = acos( q/(p*z*2.0) ) / 3.0;
      float m = cos(v);
      float n = sin(v)*1.732050808;
      vec3 t = vec3(m + m, -n - m, n - m) * z - kx;
      t = clamp( t, 0.0, 1.0 );
      vec2 qos = d + (c + b*t.x)*t.x;
      res = dot(qos,qos);
      qos = d + (c + b*t.y)*t.y;
      res = min(res,dot(qos,qos));
      qos = d + (c + b*t.z)*t.z;
      res = min(res,dot(qos,qos));
  }
  
  return sqrt( res );
}

float opSmoothSubtraction( float d1, float d2, float k ) {
  float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
  return mix( d2, -d1, h ) + k*h*(1.0-h); 
}

float opIntersection( float d1, float d2 ) {
  return max(d1,d2);
}

vec2 opSmoothUnion( float d1, float d2, float k, float label) {
  float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
  return vec2(mix( d2, d1, h ) - k*h*(1.0-h), h + label);
}

vec2 minFirst( vec2 d1, vec2 d2 )
{
	return (d1.x<d2.x) ? d1 : d2;
}

// Petal SDF
////////////////////////////////////////////////////////////////////////////////////////////
vec2 sdPetal(vec3 p, float r, float d, float angleY, float label) {
  float b = sqrt(r*r-d*d);
  mat3 transformInv = rotMatInv(vec3(-10.0 * bloom + 45.0, angleY , 5.0));
  p = transformInv * p;
  p += vec3(0.0, 0.0, -0.22);
  p = opCheapBendY(p, -1.6 * bloom + 0.3);
  p = opCheapBendX(p, 0.4 * bloom - 0.2);
  p = opCheapBendZ(p, 0.06 * bloom);
  p += vec3(0.0, -b, 0.0);
  vec2 dist = sdVesica(p.xy, r, d, label);
  return vec2(opRound(opExtrussion(p, dist.x, 0.005 ), 0.01), dist.y);
}

vec2 sdFlower(vec3 p, float r, float d, vec2 v1, vec2 v2, float label) {
  vec2 diff = v2 - v1;
  p = rotMatInv(vec3(0.0, 0.0, degrees(-atan(diff.x, diff.y)))) * p;
  p = p - vec3(0.0, 0.5, 0.0);
  vec2 dis = vec2(maxMarchLength, -1.0);
  float angle = 360.0 / float(u_Petal);
  for (int i = 0; i < u_Petal; i++) {
    dis = minFirst(dis, sdPetal(p, r, d, float(i) * angle, label));
  }
  dis = minFirst(dis, 
                vec2(opSmoothSubtraction(sdSphere(p + vec3(0., 0.5, 0.), 0.7 - 0.2 * bloom), sdSphere(p, 0.5 - 0.2 * bloom), 0.3),
                label + 100.0));
  return dis;
}

// Stem SDF
////////////////////////////////////////////////////////////////////////////////////////////
vec2 sdStem(vec3 p, vec2 v0, vec2 v1, vec2 v2, float label) {
  float d0 = opRound(opExtrussion(p, sdBezier(p.xy, v0, v1, v2), 0.001), 0.15);
  vec2 diff = v2 - v1;
  p = p - vec3(v2, 0);
  p = rotMatInv(vec3(0.0, 0.0, degrees(-atan(diff.x, diff.y)))) * p;
  float d1 = opSmoothSubtraction(sdSphere(p + vec3(0., -0.68, 0.), 0.3), sdRoundCone(p + vec3(0., 0.0, 0.), 0.2, 0.35, 0.5), 0.2);
  return opSmoothUnion(d0, d1, 0.1, label);
}

// Bounding Box
////////////////////////////////////////////////////////////////////////////////////////////
bool rayCubeCast(vec3 ori, vec3 dir, vec3 cubeCenter, vec3 cubeSize) {
  vec3 o = ori - cubeCenter;
  float t_near = -maxMarchLength;
  float t_far = maxMarchLength;
  for (int i = 0; i < 3; i++) {
    if(abs(dir[i]) < 1e-5) {
      if (o[i] < -cubeSize[i] || o[i] > cubeSize[i]) return false;
      else continue;
    }
    float t0 = (-cubeSize[i] - o[i]) / dir[i];
    float t1 = (cubeSize[i] - o[i]) / dir[i];
    if (t0 < 0.0 && t1 < 0.0) return false;
    if (t0 > t1) {
      float tt = t0;
      t0 = t1;
      t1 = tt;
    }
    if (t0 > t_near) t_near = t0;
    if (t1 < t_far) t_far = t1;
  }

  if (t_near > t_far) {
    return false;
  }
  return true;
}

struct BoundBox {
  vec3 center;
  vec3 size;
};

BoundBox bbFlower(vec3 t, float r, float d) {
  float l = sqrt(r * r - d * d) + 0.5;
  BoundBox bb;
  bb.center = t + vec3(0.0, 0.5 + l, 0.0);
  bb.size = vec3(2.0 * l);
  return bb;
}

BoundBox bbStem(vec3 t, vec2 v0, vec2 v1, vec2 v2) {
  vec2 maxCorner = max(v0, max(v1, v2)) + vec2(0.0, 0.5);
  vec2 minCorner = min(v0, min(v1, v2));
  vec2 l = maxCorner - minCorner;
  BoundBox bb;
  bb.center = vec3(t.x, (maxCorner.y + minCorner.y) / 2.0 + t.y, t.z);
  bb.size = vec3(l.x, l.y / 2.0, l.x) + vec3(0.15);
  return bb;
}

BoundBox bbCombine(BoundBox bb0, BoundBox bb1) {
  vec3 maxCorner = max(bb0.center + bb0.size, bb1.center + bb1.size);
  vec3 minCorner = min(bb0.center - bb0.size, bb1.center - bb1.size);
  BoundBox bb;
  bb.center = (maxCorner + minCorner) / 2.0;
  bb.size = (maxCorner - minCorner) / 2.0;
  return bb;
}


// Map
////////////////////////////////////////////////////////////////////////////////////////////
bool flag[4] = bool[4](true, true, true, true);

vec2 map(vec3 pos, vec3 dir)
{
  vec2 d = vec2(maxMarchLength, -1.0);
  vec3 endPoint = mix(vec3(0.0, 11.2, 0.0), vec3(-1.0, 10.0, 0.0), bloom);
  vec3 midPoint = mix(vec3(0.8, 6.4, 0.0), vec3(1.0, 6.0, 0.0), bloom);

  BoundBox bb_flower0 = bbFlower(endPoint, 2.5, 1.8);
  BoundBox bb_stem0 = bbStem(vec3(0.0, 0.0, 0.0), vec2(0.0, 0.0), midPoint.xy, endPoint.xy);
  BoundBox bb_plant0 = bbCombine(bb_flower0, bb_stem0);
  BoundBox bb_base;
  bb_base.center = vec3(0., -25., 0.);
  bb_base.size = vec3(50., 25., 50.);

  vec3 q = pos;
  if (flag[0] && rayCubeCast(pos, dir, bb_plant0.center, bb_plant0.size)) {
    if (flag[1] && rayCubeCast(pos, dir, bb_stem0.center, bb_stem0.size)) {
      d = minFirst(d, sdStem(q, vec2(0.0, 0.0), midPoint.xy, endPoint.xy, 10.0));
    } 
    else flag[1] = false;

    if (flag[2] && rayCubeCast(pos, dir, bb_flower0.center, bb_flower0.size)) {
      d = minFirst(d, sdFlower(q - endPoint, 2.5, 1.8, midPoint.xy, endPoint.xy, 100.0));
    } 
    else flag[2] = false;
  } 
  else flag[0] = false;

/*
  if (flag[3] && rayCubeCast(pos, dir, bb_base.center, bb_base.size)) {
    float dtmp = opIntersection(sdSphere(q, 50.0), sdBox(q + vec3(0, 50.0, 0.0), vec3(50.0)));
    d = dtmp < 0.0? d : minFirst(d, vec2(dtmp, 0.0));
  } 
  else flag[3] = false;
*/

//#define DRAWBOUND
#ifdef DRAWBOUND
  d = minFirst(d, vec2(sdBox(pos - bb_flower0.center, bb_flower0.size), 1000.0));
  d = minFirst(d, vec2(sdBox(pos - bb_stem0.center, bb_stem0.size), 1001.0));
  d = minFirst(d, vec2(sdBox(pos - bb_plant0.center, bb_plant0.size), 1002.0));
  //d = minFirst(d, vec2(sdBox(pos - bb_base.center, bb_base.size), 1003.0));
#endif

  return d;
}

// Normal Calculation
// Reference from http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
////////////////////////////////////////////////////////////////////////////////////////////
vec3 calcNormal(vec3 pos, vec3 dir)
{
    const float ep = 0.0001;
    vec2 e = vec2(1.0,-1.0)*0.5773;
    return normalize( e.xyy*map( pos + e.xyy*ep, dir ).x + 
					  e.yyx*map( pos + e.yyx*ep, dir ).x + 
					  e.yxy*map( pos + e.yxy*ep, dir ).x + 
					  e.xxx*map( pos + e.xxx*ep, dir ).x );
}

// Ray Marching Funtion
// Reference from https://www.shadertoy.com/view/4lyfzw
////////////////////////////////////////////////////////////////////////////////////////////
bool rayMarch(in float stepSize, in vec3 dir, out vec3 norm, out float colFlag) {
  vec2 d = map(u_Eye, dir);
  colFlag = d.y;
  float t = d.x * stepSize;
  float curLength = t;
  vec3 p = u_Eye + t * dir;
  for (int i = 0; i < maxMarchIter; i++) {
    d = map(p, dir);
    colFlag = d.y;
    t = d.x * stepSize;
    p += t * dir;
    curLength += t;
    if (curLength > maxMarchLength || t < epsilon * stepSize) {
      break;
    }
  }

  if (t < epsilon * stepSize) {
    norm = calcNormal(p, dir);
    return true;
  }
  
  colFlag = -1.0;
  return false;
}

// Ray Casting Funtion
////////////////////////////////////////////////////////////////////////////////////////////
vec3 rayCast(vec2 p) {
  vec3 focusVector = u_Ref - u_Eye;
  vec3 Right = normalize(cross(focusVector, u_Up)) * u_Dimensions.x / u_Dimensions.y;
  float len = length(focusVector);
  return normalize(focusVector + len * p.x * Right + len * p.y * u_Up);
}

// Toolbox Function
////////////////////////////////////////////////////////////////////////////////////////////
float bias(float b, float t) {
  return pow(t, log(b) / log(0.5));
}

float gain(float g, float t) {
  return (t < 0.5 ? bias(1.0 - g, 2.0 * t) / 2.0 : 1.0 - bias(1.0 - g, 2.0 - 2.0 * t) / 2.0);
}

float triangle_wave(float x, float freq) {
  return abs(fract(x * freq) - 0.5) * 2.0;
}


void main() {

  vec3 dir = rayCast(fs_Pos);
  float freq = 0.003;
  bloom = gain(0.9, triangle_wave(u_Time, freq));

  vec4 col;
  vec3 norm;
  float colFlag;
  if(rayMarch(0.4, dir, norm, colFlag)) {

    if (colFlag >= 100.0 && colFlag <= 101.0) {
      col = mix(vec4(vec3(u_Color) / 2.0 + vec3(0.5), 1.0) , u_Color, colFlag - 100.0);
      col = mix(col, vec4(1.0), bloom);
    }
    else if (colFlag >= 10.0 && colFlag <= 11.0 ) {
      col = mix(vec4(94.0, 192.0, 50.0, 255.0), vec4(76.0, 148.0, 44.0, 255.0), colFlag - 10.0) / 255.0;
    }
    else if (colFlag >= 200.0 && colFlag <= 201.0) {
      col = mix(vec4(247.0, 204.0, 74.0, 255.0), vec4(231.0, 140.0, 23.0, 255.0), colFlag - 200.0) / 255.0;
    }
    else {
      col = vec4(1.0);
    }
    float diffuseTerm = clamp(-dot(norm, normalize(lightVec)), 0.0, 1.0);
    col = vec4(col.rgb * (diffuseTerm + 0.5), 1.0);
  }
  else {
    //col = vec4(0.5 * (fs_Pos + vec2(1.0)), 0.5 * (sin(u_Time * 3.14159 * 0.01) + 1.0), 1.0);
    col = vec4(100.0, 100.0, 100.0, 255.0) /255.0;
  }

  out_Col = col;
  //out_Col = vec4(0.5 * (dir + vec3(1.0, 1.0, 1.0)), 1.0);
}
