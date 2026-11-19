export const fullscreen = `#version 300 es
void main() {
    vec2 corner = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`;

export const clouds = `#version 300 es
precision highp float;
uniform vec2 resolution;
uniform float seed, sceneScale, angle, hue;
uniform vec4 layers[5];
out vec4 pixel;

vec2 gradient(vec2 cell) {
    vec3 p = fract(vec3(cell.xyx) * vec3(.1031, .1030, .0973) + seed);
    p += dot(p, p.yzx + 33.33);
    return fract((p.xx + p.yz) * p.zy) * 2.0 - 1.0;
}

float noise(vec2 p) {
    vec2 cell = floor(p), f = fract(p);
    vec2 w = f*f*f*(f*(f*6.0-15.0)+10.0);
    float a = dot(gradient(cell), f);
    float b = dot(gradient(cell+vec2(1,0)), f-vec2(1,0));
    float c = dot(gradient(cell+vec2(0,1)), f-vec2(0,1));
    float d = dot(gradient(cell+vec2(1,1)), f-vec2(1,1));
    return 1.6 * mix(mix(a,b,w.x),mix(c,d,w.x),w.y);
}

float turbulence(vec2 p, int octaveCount, float damping) {
    float total = 0.0, amplitude = 0.57, frequency = 1.0;
    for (int i=0; i<7; i++) {
        if (i >= octaveCount) break;
        total += noise(p) * amplitude * exp(-damping*frequency*.17);
        p = mat2(1.64,-1.08,1.08,1.64)*p + vec2(13.2,19.1);
        frequency *= 1.9;
        amplitude *= 0.62;
    }
    return total;
}

vec3 hsv(float h, float s, float v) {
    vec3 color = clamp(abs(fract(h + vec3(0, 2, 1)/3.0)*6.0-3.0)-1.0,0.0,1.0);
    return v * mix(vec3(1), color, s);
}

void main() {
    vec2 screen = (gl_FragCoord.xy-.5*resolution)
                / (max(resolution.x,resolution.y)*sceneScale);
    vec2 p = mat2(cos(angle),sin(angle),-sin(angle),cos(angle))*screen;
    vec2 warped = p + vec2(.17,.12)*vec2(
        turbulence(p*3.1+17.0,3,.16), turbulence(p*3.4+32.0,3,.16));
    float disk = warped.y-.047*sin(6.0*warped.x+seed*5.0);
    float broad = turbulence(warped*vec2(7,10)+40.0,4,.08);
    vec3 light = vec3(0), transmission = vec3(1);
    float dustColumn = 0.0;
    for (int i=0; i<5; i++) {
        vec4 layer = layers[i];
        vec2 local = vec2(warped.x-layer.x, disk-layer.y + .038*broad);
        float shear = .28*sin(float(i)*2.3+seed*12.0);
        vec2 uv = vec2(local.x, local.y-shear*local.x)/layer.zw;
        float envelope = exp(-.62*pow(length(uv),2.15));
        if (envelope < .004) continue;
        vec2 cavityPos = uv+vec2(.28*broad-.34, .25+.32*broad);
        float cavityRadius = length(cavityPos*vec2(.92,1.0));
        float cavity = exp(-.73*cavityRadius*cavityRadius);
        float compression = exp(-3.8*pow(cavityRadius-1.13,2.0));
        vec2 alongFlow = vec2(uv.x+.27*sin(3.0*uv.y),uv.y+.18*uv.x*uv.x);
        float detail = turbulence(alongFlow*vec2(2.1,3.6)+17.0*float(i),7,.26-.19*compression);
        float fold = turbulence(alongFlow*vec2(2.6,6.5)-21.0*float(i),5,.09);
        float particles = smoothstep(.025,.55,fold+.37*detail);
        float density = (i==4 ? .25 : .80)*envelope*(1.0-.66*cavity)
                      *exp(1.44*detail-.57*particles);
        float opacity = 1.0-exp(-density);
        float dust = envelope*(.13+.23*cavity)*exp(.64*detail+1.45*particles);
        float core = exp(-.83*dot(uv+vec2(.38,-.23),uv+vec2(.38,-.23)));
        float rim = smoothstep(-.38,.54,.44*uv.x-.34*uv.y+.59*broad);
        float shoulder = compression*smoothstep(-.24,.36,detail)*smoothstep(.21,.60,envelope);
        float glow = .22 + .19*core + .12*shoulder + .07*detail;
        float drift = .010*sin(8.0*p.x+3.0*p.y+seed*9.0);
        float offset = .051*sin(2.35*float(i)+seed*23.0);
        vec3 fill = hsv(hue+offset+drift-.055*rim,.69,glow);
        vec3 interior = hsv(hue+offset+drift+.042,.47,glow*1.34);
        vec3 color = mix(fill,interior,smoothstep(.17,.87,core+.21*detail));
        light += transmission*opacity*color*(.88+.46*shoulder);
        vec3 extinction = mix(vec3(.86,1.,1.16),1.21-.37*color/max(max(color.r,color.g),color.b),.58);
        transmission *= (1.0-opacity)*exp(-dust*extinction);
        dustColumn += dust;
    }
    float veil = exp(-25.0*disk*disk-2.6*warped.x*warped.x);
    float grain = fract(sin(dot(gl_FragCoord.xy,vec2(12.9898,78.233)))*43758.5453)-.5;
    vec3 base = vec3(.016,.021,.036) + hsv(hue-.035,.66,.071)*veil*(.56+broad);
    pixel = vec4(max(vec3(0), light*1.65+base*transmission+grain*.007),min(dustColumn*.45,1.0));
}`;

export const display = `#version 300 es
precision highp float;
uniform sampler2D scene;
out vec4 pixel;
void main() { pixel = vec4(texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb, 1.0); }
`;

export const stars = `#version 300 es
precision highp float;
layout(location=0) in vec4 star;
layout(location=1) in float asymmetry;
uniform sampler2D scene;
uniform float sizeScale;
flat out vec3 tint;
flat out float flux, spread, phase;
void main() {
    gl_Position = vec4(star.xy,0,1);
    vec4 gas = texture(scene,star.xy*.5+.5);
    float behind = star.w;
    tint = mix(vec3(.86,.89,1.), normalize(max(gas.rgb,vec3(.018)))*1.45, .39*behind);
    flux = star.z*exp(-2.8*gas.a*behind);
    spread = flux > 1.4 ? 2.0 : 1.0;
    phase = asymmetry;
    gl_PointSize = (flux > 1.4 ? 12. : 5.) * sizeScale;
}
`;

export const starLight = `#version 300 es
precision highp float;
flat in vec3 tint;
flat in float flux, spread, phase;
out vec4 pixel;
void main() {
    vec2 p = gl_PointCoord*2.0-1.0;
    float angle = atan(p.y,p.x);
    float r2 = dot(p,p)*(1.0+.10*cos(3.0*angle+phase));
    float core = exp(-r2*16.0*spread);
    float wing = .108*exp(-r2*4.7) + .018*exp(-r2*2.2);
    float alpha = 1.0-exp(-flux*(core+wing));
    vec3 rgb = mix(tint,vec3(.98,.98,1.),core);
    pixel = vec4(rgb*alpha,alpha);
}`;
