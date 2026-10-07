/* core.js - indexes, severity lookup, 13 network features, gradient-boosting inference */
const N=D.d,S=D.s,LN=['Unknown','Minor','Moderate','Major'],LC=['--unk','--min','--mod','--maj'];
const M=new Map(),nb=D.d.map(()=>new Map());let total=0;const cnt=[0,0,0,0];
for(const k in D.p){const a=+k,v=D.p[k];for(let t=0;t<v.length;t+=2){M.set(a*2048+v[t],v[t+1]);nb[a].set(v[t],v[t+1]);nb[v[t]].set(a,v[t+1]);cnt[v[t+1]]++;total++}}
const lv=(i,j)=>M.get(Math.min(i,j)*2048+Math.max(i,j));
const adv=['No severity has been assigned in DDInter. Review the primary literature.','Minor: limited clinical effect. Monitoring is usually enough.','Moderate: may need monitoring or dose adjustment. Weigh benefit against risk.','Major: potentially serious. Avoid the combination or use only with close supervision.'];
// gradient-boosting model trained offline on interaction-network features
function feats(i,j){const own=nb[i].get(j),o=own!==undefined;
 const st=x=>{const c=S[x].slice();if(o)c[3-own]--;const tot=c[0]+c[1]+c[2]+c[3],g=c[0]+c[1]+c[2];return[tot,c[0],g?c[0]/g:0,g?c[1]/g:0,g?c[2]/g:0]};
 const a=st(i),b=st(j);let A=nb[i],B=nb[j];if(A.size>B.size)[A,B]=[B,A];let cnt=0,ng=0,b3=0,e3=0,sl=0;
 for(const[k,la]of A){const lb=B.get(k);if(lb===undefined)continue;cnt++;if(la>0&&lb>0){ng++;sl+=la+lb}if(la==3&&lb==3)b3++;if(la==3||lb==3)e3++}
 const un=nb[i].size+nb[j].size-2*o-cnt,mu=a[1]+b[1]-b3;
 return[Math.log1p(Math.min(a[0],b[0])),Math.log1p(Math.max(a[0],b[0])),(a[2]+b[2])/2,Math.max(a[2],b[2]),(a[3]+b[3])/2,(a[4]+b[4])/2,un>0?cnt/un:0,mu>0?b3/mu:0,Math.log1p(cnt),ng?sl/(2*ng)/3:0,cnt?b3/cnt:0,cnt?e3/cnt:0,cnt?ng/cnt:0]}
function gbp(x){const r=MD.init.slice();for(const row of MD.trees)for(let c=0;c<3;c++){const[f,th,l,rr,v]=row[c];let n=0;while(l[n]!==-1)n=x[f[n]]<=th[n]?l[n]:rr[n];r[c]+=MD.lr*v[n]}
 const m=Math.max(...r),e=r.map(v=>Math.exp(v-m)),s=e[0]+e[1]+e[2];return e.map(v=>v/s)}
const predict=(i,j)=>gbp(feats(i,j));
