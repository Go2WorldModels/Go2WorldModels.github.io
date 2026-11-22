(() => {
  "use strict";
  const el = id => document.querySelector("#pose-" + id);
  const canvas = el("canvas"), ctx = canvas.getContext("2d");
  const sources = ["telemetry", "neural", "mujoco"];
  const colors = {telemetry: "#272b2a", neural: "#c14c18", mujoco: "#007b83"};
  const names = ["Front left", "Front right", "Rear left", "Rear right"];
  const clock = el("time"), seek = el("seek"), play = el("play"), trial = el("trial");
  const layers = [...document.querySelectorAll(".pose-layers input")];
  const presets = {oblique: [1.04, .27], side: [Math.PI/2, .025], front: [0, .025], top: [Math.PI/2, Math.PI/2]};
  let az, elev, data, frame = 0, playing = false, epoch = 0, startFrame = 0, requestId = 0, preview = true;
  const add = (a,b) => a.map((x,i) => x+b[i]);
  const sub = (a,b) => a.map((x,i) => x-b[i]);
  const scale = (a,k) => a.map(x => x*k);
  const norm = a => Math.hypot(...a);
  const unit = a => scale(a,1/norm(a));
  const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
  const dot = (a,b) => a.reduce((v,x,i) => v+x*b[i],0);
  const hidden = source => !layers.find(x => x.value===source).checked;
  function draw() {
    if (!data) return;
    const w = canvas.clientWidth, h = canvas.clientHeight, dpr = Math.min(2,devicePixelRatio);
    if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
      canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
    }
    ctx.setTransform(dpr,0,0,dpr,0,0);
    ctx.fillStyle="#f6f4ed"; ctx.fillRect(0,0,w,h);
    const compact = w<580, px = Math.min(w*1.12,(h-(compact?80:90))*1.38);
    const side = [-Math.sin(az),Math.cos(az),0];
    const vertical = [-Math.cos(az)*Math.sin(elev),-Math.sin(az)*Math.sin(elev),Math.cos(elev)];
    const depth = cross(side,vertical);
    const base = data.base_xyz[frame], foot = +el("foot").value*4+3;
    const focus = [base[0],base[1],.28];
    function project(a) {
      const p=sub(a,focus);
      return [w*.5+px*dot(p,side), h*.54-px*dot(p,vertical)];
    }
    function line(points,color,width,dash=[]) {
      ctx.beginPath();
      points.forEach((p,i) => {const [x,y]=project(p); if(i) ctx.lineTo(x,y); else ctx.moveTo(x,y);});
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.lineJoin="round";ctx.lineCap="round";ctx.stroke();ctx.setLineDash([]);
    }
    function circle(p,r,color,filled) {
      const [x,y]=project(p);ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
      if(filled) {ctx.fillStyle=color;ctx.fill();} else {ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
    }
    const grid=.6096, gx=Math.floor(base[0]/grid), gy=Math.floor(base[1]/grid);
    for(let i=-8;i<=8;i++){
      const x=(gx+i)*grid,y=(gy+i)*grid;
      line([[x,(gy-9)*grid,0],[x,(gy+9)*grid,0]],"#d6d2c4",1);
      line([[(gx-9)*grid,y,0],[(gx+9)*grid,y,0]],"#d6d2c4",1);
    }
    const poses=Object.fromEntries(sources.map(s=>[s,data.poses[s][frame].map(p=>add(p,base))]));
    const factor=compact?.78:1;
    const real=poses.telemetry;
    if(!hidden("telemetry")){
      const ex=unit(sub(real[0],real[8])),ey=unit(sub(real[0],real[4])),ez=unit(cross(ex,ey));
      const transform=([x,y,z])=>add(base,add(scale(ex,x),add(scale(ey,y),scale(ez,z))));
      // A translucent schematic trunk gives the link frames a readable orientation.
      const body=[[-.20,-.065,-.047],[-.20,.065,-.047],[-.20,.065,.05],[-.20,-.065,.05],
        [.19,-.065,-.047],[.19,.065,-.047],[.19,.065,.05],[.19,-.065,.05]].map(transform);
      const faces=[[0,1,2,3],[4,7,6,5],[0,4,5,1],[2,6,7,3],[0,3,7,4],[1,5,6,2]];
      faces.sort((a,b)=>a.reduce((s,k)=>s+dot(body[k],depth),0)-b.reduce((s,k)=>s+dot(body[k],depth),0));
      for(const face of faces){
        ctx.beginPath();face.forEach((n,i)=>{const [x,y]=project(body[n]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});
        ctx.closePath();ctx.fillStyle="rgba(86,92,87,.25)";ctx.fill();
      }
      line([transform([.16,0,.01]),transform([.277,0,.01])],"#626761",13*factor);
      circle(transform([.278,0,.01]),5*factor,"#252a28",true);
    }
    for(const source of sources){
      if(hidden(source)) continue;
      const p=poses[source],col=colors[source];
      const width={telemetry:7,neural:3.5,mujoco:2.6}[source]*factor;
      const dash=source==="mujoco"?[6*factor,4*factor]:[];
      const trail=[];
      for(let i=Math.max(0,frame-20);i<=frame;i++){
        trail.push(add(data.base_xyz[i],data.poses[source][i][foot]));
      }
      ctx.globalAlpha=.63;line(trail,col,2.1*factor,dash);ctx.globalAlpha=1;
      line([p[0],p[4],p[12],p[8],p[0]],col,width*.65,dash);
      const legs=[0,4,8,12].sort((a,b)=>dot(p[a],depth)-dot(p[b],depth));
      for(const j of legs){
        line(p.slice(j,j+4),col,width,dash);
        circle(p[j+2],width*.8,col,source==="telemetry");
        circle(p[j+3],(source==="mujoco"?4:source==="neural"?5.5:6.5)*factor,col,source==="telemetry");
      }
      circle(p[foot],(source==="mujoco"?7:source==="neural"?9:11)*factor,col,false);
    }
    const size=compact?12:16;
    ctx.font='600 '+size+'px system-ui, sans-serif';ctx.textBaseline="top";
    ctx.fillStyle="#24251e";ctx.fillText(data.label,14,10);
    const timing=(frame*data.dt_s).toFixed(2)+" s";
    ctx.textAlign="right";ctx.fillText(timing,w-14,10);ctx.textAlign="left";
    let left=14;
    for(const [i,label] of (compact?["Go2","Neural","MuJoCo"]:["Go2 telemetry","120M world model","MuJoCo"]).entries()){
      if(hidden(sources[i])) continue;
      ctx.fillStyle=colors[sources[i]];ctx.fillText(label,left,compact?31:36);
      left+=ctx.measureText(label).width+(compact?16:25);
    }
    const distances=sources.slice(1).map(s=>norm(sub(poses[s][foot],real[foot]))*100);
    ctx.font=(compact?11:15)+'px system-ui, sans-serif';ctx.fillStyle="#34372f";
    const footLabel=names[foot>>2]+" foot";
    const footDetail="Neural "+distances[0].toFixed(1)+" cm  ·  MuJoCo "+distances[1].toFixed(1)+" cm";
    ctx.fillText(footLabel+" separation   "+footDetail,14,h-27);
    el("readout").textContent="Onboard planar displacement "+Math.hypot(base[0],base[1]).toFixed(2)+" m   ·   Reported body height "+(base[2]*100).toFixed(1)+" cm";
    clock.textContent=timing+" / "+((data.base_xyz.length-1)*data.dt_s).toFixed(2)+" s";
    seek.value=frame;
  }
  function stop() {playing=false; play.textContent="Play";}
  function tick(now){
    if(!playing||!data)return;
    frame=Math.min(data.base_xyz.length-1,startFrame+Math.floor((now-epoch)*.001*Number(el("speed").value)/data.dt_s));
    draw();
    if(frame===data.base_xyz.length-1){stop();play.textContent="Replay";}else requestAnimationFrame(tick);
  }
  function setView(){
    [az,elev]=presets[el("view").value]||presets.oblique;draw();
  }
  async function load(){
    stop();data=null;const id=++requestId, key=trial.value;
    play.disabled=true;seek.disabled=true;clock.textContent="Loading…";
    const file="assets/"+key+"-poses.json";
    try{
      const response=await fetch(file);
      if(!response.ok)throw Error("missing data");
      const next=await response.json();
      if(id!==requestId)return;
      if(next.trial!==key || !next.base_xyz.length)throw Error("invalid data");
      data=next;seek.max=data.base_xyz.length-1;frame=Math.round(Number(seek.max)*.43);
      preview=true;play.disabled=false;seek.disabled=false;el("data").href=file;el("movie").href="assets/"+key+"-overlay.mp4";draw();
    }catch(error){if(id===requestId){clock.textContent="Replay unavailable";el("readout").textContent="Reload to retry loading the poses.";}}
  }
  play.addEventListener("click",()=>{
    if(playing){stop();return;}if(!data)return;
    if(preview||frame===data.base_xyz.length-1)frame=0;
    preview=false;playing=true;startFrame=frame;epoch=performance.now();play.textContent="Pause";requestAnimationFrame(tick);
  });
  seek.addEventListener("input",()=>{stop();preview=false;frame=+seek.value;draw();});
  el("speed").addEventListener("change",()=>{if(playing){epoch=performance.now();startFrame=frame;}});
  el("foot").addEventListener("change",draw);el("view").addEventListener("change",setView);
  trial.addEventListener("change",load);layers.forEach(x=>x.addEventListener("change",draw));
  function orbit(dx,dy){az-=dx;elev=Math.max(-1.4,Math.min(Math.PI/2,elev+dy));el("view").value="custom";draw();}
  el("view").add(new Option("Orbit","custom",false,false));el("view").lastChild.disabled=true;
  let drag=null;
  canvas.addEventListener("pointerdown",e=>{if(drag)return;canvas.setPointerCapture(e.pointerId);drag=[e.clientX,e.clientY,e.pointerId];});
  canvas.addEventListener("pointermove",e=>{if(drag&&e.pointerId===drag[2]){orbit((e.clientX-drag[0])*.009,(e.clientY-drag[1])*.008);drag=[e.clientX,e.clientY,e.pointerId];}});
  canvas.addEventListener("lostpointercapture",()=>{drag=null;});
  canvas.addEventListener("keydown",e=>{const dirs={ArrowRight:[.1,0],ArrowLeft:[-.1,0],ArrowUp:[0,.1],ArrowDown:[0,-.1]};
    if(dirs[e.key]){e.preventDefault();orbit(...dirs[e.key]);}});
  const requested=new URLSearchParams(location.search).get("trial");
  if([...trial.options].some(o=>o.value===requested))trial.value=requested;
  new ResizeObserver(draw).observe(canvas);setView();load();
})();
