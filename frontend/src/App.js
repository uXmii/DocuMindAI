import React, { useState, useEffect, useRef } from 'react';

/* ─── Fonts ──────────────────────────────────────────────────────────────── */
const _fl = document.createElement('link');
_fl.rel  = 'stylesheet';
_fl.href = 'https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&display=swap';
document.head.appendChild(_fl);

/* ─── Global CSS ─────────────────────────────────────────────────────────── */
const _gs = document.createElement('style');
_gs.textContent = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Geist', -apple-system, sans-serif;
    background: #f5f5f4;
    color: #1c1917;
    -webkit-font-smoothing: antialiased;
  }
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: #d6d3d1; border-radius: 10px; }
  ::-webkit-scrollbar-thumb:hover { background: #a8a29e; }
  @keyframes fadeUp   { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
  @keyframes pulseDot { 0%,80%,100%{transform:scale(0.5);opacity:0.35} 40%{transform:scale(1);opacity:1} }
  @keyframes spin     { to{transform:rotate(360deg)} }
  @keyframes scoreIn  { from{width:0%} }
  .fade-up { animation: fadeUp 0.28s cubic-bezier(.16,1,.3,1) both; }
  textarea { line-height: 1.5; }
`;
document.head.appendChild(_gs);

/* ─── Design tokens ──────────────────────────────────────────────────────── */
const C = {
  // backgrounds
  bg:        '#f5f5f4',
  surface:   '#ffffff',
  surfaceAlt:'#fafaf9',
  hover:     '#f5f5f4',
  // borders
  border:    '#e7e5e4',
  borderMid: '#d6d3d1',
  // text
  text:      '#1c1917',
  textMid:   '#57534e',
  textLow:   '#a8a29e',
  textXLow:  '#d6d3d1',
  // accent blue
  blue:      '#2563eb',
  blueLight: '#dbeafe',
  blueBorder:'#bfdbfe',
  // semantic
  green:     '#16a34a',
  greenLight:'#dcfce7',
  amber:     '#d97706',
  amberLight:'#fef3c7',
  red:       '#dc2626',
  redLight:  '#fee2e2',
  violet:    '#7c3aed',
  violetLight:'#ede9fe',
  cyan:      '#0891b2',
  cyanLight: '#cffafe',
};

/* ─── Helpers ────────────────────────────────────────────────────────────── */
const pct  = v => v != null ? `${(v*100).toFixed(0)}%` : '—';
const msf  = v => v != null ? `${(v*1000).toFixed(0)}ms` : '—';
const fmt3 = v => v != null ? v.toFixed(3) : '—';

/* ─── Icon paths ─────────────────────────────────────────────────────────── */
const PATHS = {
  logo:    'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  send:    'M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z',
  upload:  'M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12',
  refresh: 'M23 4v6h-6M1 20v-6h6M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15',
  trash:   'M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6',
  zap:     'M13 2L3 14h9l-1 8 10-12h-9l1-8z',
  brain:   'M12 5a3 3 0 01-5.98-.36A3 3 0 018 2a3 3 0 013 3zm0 0a3 3 0 005.98-.36A3 3 0 0016 2a3 3 0 00-3 3M12 5v14M8 9H4a2 2 0 000 4h4M16 9h4a2 2 0 010 4h-4',
  search:  'M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z',
  chart:   'M18 20V10M12 20V4M6 20v-6',
  eye:     'M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8zM12 9a3 3 0 100 6 3 3 0 000-6z',
  x:       'M18 6L6 18M6 6l12 12',
  check:   'M20 6L9 17l-5-5',
  info:    'M12 22a10 10 0 100-20 10 10 0 000 20zm0-9v4m0-8h.01',
  file:    'M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8zM14 2v6h6',
  compare: 'M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01',
  trophy:  'M8 21h8M12 17v4M17 3H7v8a5 5 0 0010 0V3z',
  sparkle: 'M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6z',
  layers:  'M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5',
  cpu:     'M18 4H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2zM9 9h6v6H9z',
  repeat:  'M17 1l4 4-4 4M3 11V9a4 4 0 014-4h14M7 23l-4-4 4-4M21 13v2a4 4 0 01-4 4H3',
  down:    'M6 9l6 6 6-6',
  up:      'M18 15l-6-6-6 6',
  globe:   'M12 2a10 10 0 100 20A10 10 0 0012 2zM2 12h20M12 2a15.3 15.3 0 010 20M12 2a15.3 15.3 0 000 20',
  warn:    'M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0zM12 9v4M12 17h.01',
  dot:     'M12 12m-2 0a2 2 0 100 4 2 2 0 000-4z',
};
const Ico = ({ n, s=16, c='currentColor', st={} }) => (
  <svg width={s} height={s} viewBox="0 0 24 24" fill="none"
    stroke={c} strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" style={st}>
    <path d={PATHS[n]||PATHS.info}/>
  </svg>
);

/* ─── Primitives ─────────────────────────────────────────────────────────── */
const Tag = ({ children, color=C.blue, bg=C.blueLight, border=C.blueBorder }) => (
  <span style={{
    display:'inline-flex', alignItems:'center', gap:3,
    padding:'2px 7px', borderRadius:5,
    fontSize:10.5, fontWeight:600, letterSpacing:'0.02em',
    color, background:bg, border:`1px solid ${border}`,
    fontFamily:"'Geist Mono',monospace", whiteSpace:'nowrap',
  }}>{children}</span>
);

const Bar = ({ v=0, color=C.blue, h=3 }) => (
  <div style={{height:h,background:C.border,borderRadius:99,overflow:'hidden'}}>
    <div style={{
      height:'100%', width:`${Math.min(v*100,100)}%`,
      background:color, borderRadius:99,
      animation:'scoreIn 0.6s cubic-bezier(.16,1,.3,1) both',
    }}/>
  </div>
);

const Divider = ({m='6px 0'}) => (
  <div style={{height:1,background:C.border,margin:m}}/>
);

const Spin = ({s=13,c=C.blue}) => (
  <div style={{width:s,height:s,borderRadius:'50%',border:`2px solid ${c}30`,
    borderTopColor:c,animation:'spin 0.75s linear infinite',flexShrink:0}}/>
);

/* ─── Method / source config ─────────────────────────────────────────────── */
const M = {
  vector: { label:'Semantic', color:C.blue,   bg:C.blueLight,   icon:'search' },
  bm25:   { label:'Keyword',  color:C.green,  bg:C.greenLight,  icon:'layers' },
  hybrid: { label:'Hybrid',   color:C.violet, bg:C.violetLight, icon:'zap'    },
};
const SRC = {
  vision_llm:{ label:'Vision AI', color:C.violet, bg:C.violetLight, icon:'sparkle' },
  tesseract: { label:'OCR',       color:C.amber,  bg:C.amberLight,  icon:'eye'     },
  table:     { label:'Table',     color:C.cyan,   bg:C.cyanLight,   icon:'chart'   },
  text:      { label:'Text',      color:C.textMid,bg:C.hover,       icon:'file'    },
};

/* ══════════════════════════════════════════════════════════════════════════
   APP
══════════════════════════════════════════════════════════════════════════ */
const API = 'http://localhost:5000';

export default function App() {
  const [msgs,    setMsgs]    = useState([]);
  const [input,   setInput]   = useState('');
  const [loading, setLoading] = useState({upload:false,query:false,eval:false,consist:false});
  const [stats,   setStats]   = useState(null);
  const [mode,    setMode]    = useState('hybrid');
  const [agent,   setAgent]   = useState(false);
  const [openAgt, setOpenAgt] = useState({});
  const [modal,   setModal]   = useState(null);
  const endRef = useRef(null);
  const taRef  = useRef(null);

  useEffect(()=>{ loadStats(); },[]);
  useEffect(()=>{ endRef.current?.scrollIntoView({behavior:'smooth'}); },[msgs,loading.query]);

  const loadStats = async () => {
    try { const r=await fetch(`${API}/stats`); setStats(await r.json()); } catch{}
  };
  const setL = (k,v) => setLoading(p=>({...p,[k]:v}));
  const post = (url,body) => fetch(`${API}${url}`,{
    method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)
  }).then(r=>r.json());
  const addMsg = m => setMsgs(p=>[...p,m]);

  const onUpload = async e => {
    const f=e.target.files[0]; if(!f) return;
    setL('upload',true);
    const fd=new FormData(); fd.append('file',f);
    try {
      const d=await fetch(`${API}/upload`,{method:'POST',body:fd}).then(r=>r.json());
      if(d.success){
        const {text=0,ocr=0,tables=0}=d.chunk_breakdown||{};
        addMsg({t:'sys',text:`"${d.filename}" indexed — ${d.chunks_created} chunks`,
          up:{text,ocr,tables,vis:d.vision_used||ocr>0,total:d.collection_size}});
        loadStats();
      } else addMsg({t:'err',text:`Upload failed: ${d.error}`});
    } catch(err){ addMsg({t:'err',text:err.message}); }
    finally { setL('upload',false); e.target.value=''; }
  };

  const onQuery = async () => {
    if(!input.trim()||loading.query) return;
    const q=input.trim(); setInput(''); setL('query',true);
    if(taRef.current){taRef.current.style.height='auto';}
    addMsg({t:'user',text:q});
    try {
      const d=await post('/query',{question:q,top_k:5,search_mode:mode});
      addMsg(d.success
        ?{t:'ai',text:d.answer,srcs:d.sources,ctx:d.context_used,
          sm:d.search_mode,st:d.search_time,tt:d.total_time,clip:d.clip_used}
        :{t:'err',text:`Query failed: ${d.error}`});
    } catch(err){ addMsg({t:'err',text:err.message}); }
    finally{ setL('query',false); }
  };

  const onAgent = async () => {
    if(!input.trim()||loading.query) return;
    const q=input.trim(); setInput(''); setL('query',true);
    if(taRef.current){taRef.current.style.height='auto';}
    addMsg({t:'user',text:q});
    try {
      const d=await post('/query/agentic',{question:q,max_iterations:2});
      addMsg(d.success
        ?{t:'ai',text:d.answer,srcs:d.sources,meta:d.metadata,
          conf:d.confidence,qs:d.quality_score,isAgent:true}
        :{t:'err',text:`Agent failed: ${d.error}`});
    } catch(err){ addMsg({t:'err',text:err.message}); }
    finally{ setL('query',false); }
  };

  const onCompare  = async () => {
    if(!input.trim()) return; setL('query',true);
    try{ const d=await post('/compare',{question:input.trim(),top_k:5});
      setModal({type:'compare',data:d}); } catch{}
    finally{ setL('query',false); }
  };
  const onEval = async () => {
    if(!input.trim()||loading.eval) return; setL('eval',true);
    try{ const d=await post('/evaluate/single',{question:input.trim()});
      setModal({type:'eval',data:d}); } catch(err){ addMsg({t:'err',text:err.message}); }
    finally{ setL('eval',false); }
  };
  const onConsist = async () => {
    if(!input.trim()||loading.consist) return; setL('consist',true);
    try{ const d=await post('/evaluate/consistency',{question:input.trim(),n_runs:3,mode:'hybrid'});
      setModal({type:'consist',data:d}); } catch(err){ addMsg({t:'err',text:err.message}); }
    finally{ setL('consist',false); }
  };
  const onReset = async () => {
    if(!window.confirm('Reset all indexed documents?')) return;
    try{ const d=await fetch(`${API}/reset`,{method:'POST'}).then(r=>r.json());
      if(d.success){setMsgs([{t:'sys',text:'Collection reset.'}]);loadStats();} } catch{}
  };

  const onKey = e => {
    if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();agent?onAgent():onQuery();}
  };
  const canSend = !loading.query && input.trim() && stats?.total_chunks>0;

  return (
    <div style={{display:'flex',flexDirection:'column',height:'100vh',background:C.bg,overflow:'hidden'}}>

      {/* ── HEADER ── */}
      <header style={{
        display:'flex',alignItems:'center',justifyContent:'space-between',
        padding:'0 20px',height:54,flexShrink:0,
        background:C.surface,borderBottom:`1px solid ${C.border}`,
        boxShadow:'0 1px 3px rgba(0,0,0,0.04)',
      }}>
        <div style={{display:'flex',alignItems:'center',gap:10}}>
          <div style={{
            width:30,height:30,borderRadius:8,
            background:'linear-gradient(135deg,#1d4ed8,#7c3aed)',
            display:'flex',alignItems:'center',justifyContent:'center',
          }}>
            <Ico n="logo" s={14} c="#fff"/>
          </div>
          <div>
            <div style={{fontSize:14,fontWeight:700,color:C.text,letterSpacing:'-0.02em'}}>DocuMind</div>
            <div style={{fontSize:9.5,color:C.textLow,fontFamily:"'Geist Mono',monospace",letterSpacing:'0.04em',marginTop:0.5}}>
              {agent?'AGENTIC · MULTIMODAL RAG':'HYBRID · MULTIMODAL RAG'}
            </div>
          </div>
        </div>

        {stats && (
          <div style={{display:'flex',gap:6,alignItems:'center'}}>
            <HPill label="Chunks"  val={stats.total_chunks}    />
            {stats.bm25_indexed       && <HPill label="BM25"   val="Active" color={C.green}  />}
            {stats.agentic_enabled    && <HPill label="Agent"  val="Ready"  color={C.blue}   />}
            {stats.multimodal_enabled && <HPill label="Vision" val="On"     color={C.violet} />}
          </div>
        )}
      </header>

      <div style={{display:'flex',flex:1,overflow:'hidden'}}>

        {/* ── SIDEBAR ── */}
        <aside style={{
          width:240,flexShrink:0,display:'flex',flexDirection:'column',
          background:C.surface,borderRight:`1px solid ${C.border}`,
          overflowY:'auto',padding:'12px 0',
        }}>
          <Sec label="Document" icon="file">
            <label style={{display:'block',cursor:'pointer'}}>
              <div style={{
                display:'flex',alignItems:'center',justifyContent:'center',gap:7,
                padding:'8px 0',borderRadius:8,
                background:loading.upload?C.hover:'linear-gradient(135deg,#1d4ed8,#7c3aed)',
                color:loading.upload?C.textMid:'#fff',fontSize:12.5,fontWeight:600,
                cursor:loading.upload?'wait':'pointer',
                boxShadow:loading.upload?'none':'0 2px 8px rgba(37,99,235,0.25)',
                transition:'all 0.15s',
              }}>
                {loading.upload?<><Spin s={11} c={C.textMid}/><span>Processing…</span></>
                  :<><Ico n="upload" s={13} c="#fff"/><span>Upload PDF</span></>}
              </div>
              <input type="file" accept=".pdf" onChange={onUpload} disabled={loading.upload} style={{display:'none'}}/>
            </label>
            <p style={{fontSize:10.5,color:C.textLow,textAlign:'center',marginTop:5}}>PDF only · max 16 MB</p>
            {stats?.multimodal_enabled && <CapRow icon="sparkle" label="Gemini Vision" color={C.violet}/>}
            {stats?.agentic_enabled    && <CapRow icon="brain"   label="LLM Judge"    color={C.blue}  />}
          </Sec>

          <Divider m="10px 0"/>

          <Sec label="Query Mode" icon="cpu">
            <AgentToggle val={agent} onChange={setAgent}/>
            {!agent ? (
              <div style={{display:'flex',flexDirection:'column',gap:2,marginTop:8}}>
                {Object.entries(M).map(([k,v])=>(
                  <MBtn key={k} active={mode===k} cfg={v} onClick={()=>setMode(k)}/>
                ))}
              </div>
            ) : (
              <div style={{
                marginTop:8,padding:'8px 10px',borderRadius:7,
                background:C.violetLight,border:`1px solid ${C.border}`,
                fontSize:11,color:C.violet,lineHeight:1.6,
              }}>
                Agent auto-selects strategy per question
              </div>
            )}
          </Sec>

          <Divider m="10px 0"/>

          <Sec label="Capabilities" icon="sparkle">
            <FeatureList agent={agent}/>
          </Sec>

          <div style={{flex:1}}/>
          <Divider m="10px 0"/>

          <Sec label="System" icon="info">
            <SBtn icon="refresh" label="Refresh Stats"    onClick={loadStats}/>
            <SBtn icon="trash"   label="Reset Collection" onClick={onReset} danger/>
          </Sec>
        </aside>

        {/* ── CHAT ── */}
        <main style={{flex:1,display:'flex',flexDirection:'column',overflow:'hidden',background:C.bg}}>
          <div style={{flex:1,overflowY:'auto',padding:'20px 24px',display:'flex',flexDirection:'column',gap:14}}>
            {msgs.length===0
              ? <Welcome agent={agent}/>
              : msgs.map((m,i)=>(
                <Bubble key={i} m={m} i={i}
                  open={openAgt}
                  toggle={i=>setOpenAgt(p=>({...p,[i]:!p[i]}))}
                  onSrc={s=>setModal({type:'src',data:s})}
                />
              ))
            }
            {loading.query&&<Typing/>}
            <div ref={endRef}/>
          </div>

          {/* ── INPUT BAR ── */}
          <div style={{
            borderTop:`1px solid ${C.border}`,background:C.surface,
            padding:'12px 24px 16px',
          }}>
            <div style={{display:'flex',gap:5,marginBottom:9,flexWrap:'wrap'}}>
              {!agent&&(
                <AChip icon="compare" label="Compare" onClick={onCompare}
                  disabled={!canSend||loading.query}/>
              )}
              <AChip icon="chart"  label="Evaluate"   onClick={onEval}
                disabled={!input.trim()||loading.eval||!stats?.total_chunks}
                loading={loading.eval} color={C.violet}/>
              <AChip icon="repeat" label="Consistency" onClick={onConsist}
                disabled={!input.trim()||loading.consist||!stats?.total_chunks}
                loading={loading.consist} color={C.cyan}/>
            </div>

            <div style={{display:'flex',gap:9,alignItems:'flex-end'}}>
              <textarea ref={taRef} rows={1} value={input}
                onChange={e=>{
                  setInput(e.target.value);
                  e.target.style.height='auto';
                  e.target.style.height=Math.min(e.target.scrollHeight,120)+'px';
                }}
                onKeyDown={onKey}
                placeholder={agent?'Ask anything — agent handles the rest…':'Ask about your documents…'}
                disabled={loading.query||!stats?.total_chunks}
                style={{
                  flex:1,resize:'none',overflow:'hidden',
                  padding:'10px 14px',borderRadius:10,
                  background:C.surfaceAlt,
                  border:`1.5px solid ${C.border}`,
                  color:C.text,fontSize:13.5,
                  fontFamily:"'Geist',sans-serif",
                  outline:'none',transition:'border-color 0.15s',
                }}
                onFocus={e=>e.target.style.borderColor=C.blue}
                onBlur={e=>e.target.style.borderColor=C.border}
              />
              <button onClick={agent?onAgent:onQuery} disabled={!canSend} style={{
                width:42,height:42,borderRadius:10,border:'none',flexShrink:0,
                cursor:canSend?'pointer':'not-allowed',
                background:canSend
                  ?(agent?'linear-gradient(135deg,#7c3aed,#4f46e5)':'linear-gradient(135deg,#1d4ed8,#2563eb)')
                  :C.hover,
                display:'flex',alignItems:'center',justifyContent:'center',
                boxShadow:canSend?'0 2px 8px rgba(37,99,235,0.25)':'none',
                transition:'all 0.15s',
              }}>
                <Ico n="send" s={15} c={canSend?'#fff':C.textLow}/>
              </button>
            </div>
          </div>
        </main>
      </div>

      {/* ── MODALS ── */}
      {modal?.type==='src'     && <MWrap title="Source Documents"     icon="eye"     onClose={()=>setModal(null)}><SrcsBody  srcs={modal.data}/></MWrap>}
      {modal?.type==='compare' && <MWrap title="Method Comparison"    icon="compare" onClose={()=>setModal(null)} wide subtitle={modal.data?.question}><CompBody   data={modal.data}/></MWrap>}
      {modal?.type==='eval'    && <MWrap title="Evaluation Results"   icon="chart"   onClose={()=>setModal(null)} wide subtitle={modal.data?.question}><EvalBody   data={modal.data}/></MWrap>}
      {modal?.type==='consist' && <MWrap title="Consistency Analysis" icon="repeat"  onClose={()=>setModal(null)} subtitle={modal.data?.question}><ConsBody   data={modal.data}/></MWrap>}
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════════════
   SIDEBAR
════════════════════════════════════════════════════════════════════════════ */
const Sec = ({label,icon,children}) => (
  <div style={{padding:'0 12px 2px'}}>
    <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:9}}>
      <Ico n={icon} s={11} c={C.textLow}/>
      <span style={{fontSize:9.5,fontWeight:700,color:C.textLow,
        letterSpacing:'0.07em',textTransform:'uppercase'}}>{label}</span>
    </div>
    {children}
  </div>
);

const HPill = ({label,val,color=C.textMid}) => (
  <div style={{display:'flex',alignItems:'center',gap:5,padding:'4px 8px',
    borderRadius:6,background:C.surfaceAlt,border:`1px solid ${C.border}`}}>
    <span style={{fontSize:10,color:C.textLow}}>{label}</span>
    <span style={{fontSize:10,fontWeight:700,color,fontFamily:"'Geist Mono',monospace"}}>{val}</span>
  </div>
);

const CapRow = ({icon,label,color}) => (
  <div style={{display:'flex',alignItems:'center',gap:6,marginTop:6}}>
    <div style={{width:5,height:5,borderRadius:'50%',background:color,
      boxShadow:`0 0 4px ${color}80`}}/>
    <Ico n={icon} s={11} c={color}/>
    <span style={{fontSize:11,color,fontWeight:500}}>{label}</span>
  </div>
);

const AgentToggle = ({val,onChange}) => (
  <button onClick={()=>onChange(!val)} style={{
    width:'100%',display:'flex',alignItems:'center',justifyContent:'space-between',
    padding:'8px 10px',borderRadius:8,border:`1.5px solid ${val?C.violet:C.border}`,
    background:val?C.violetLight:C.surfaceAlt,cursor:'pointer',transition:'all 0.15s',
  }}>
    <div style={{display:'flex',alignItems:'center',gap:7}}>
      <Ico n={val?'brain':'cpu'} s={13} c={val?C.violet:C.textMid}/>
      <span style={{fontSize:12,fontWeight:600,color:val?C.violet:C.textMid}}>
        {val?'Agentic Mode':'Manual Mode'}
      </span>
    </div>
    <div style={{width:28,height:16,borderRadius:99,background:val?C.violet:C.borderMid,
      position:'relative',transition:'background 0.2s',flexShrink:0}}>
      <div style={{position:'absolute',top:2.5,left:val?13:2.5,width:11,height:11,
        borderRadius:'50%',background:'#fff',transition:'left 0.2s',
        boxShadow:'0 1px 3px rgba(0,0,0,0.2)'}}/>
    </div>
  </button>
);

const MBtn = ({active,cfg,onClick}) => (
  <button onClick={onClick} style={{
    display:'flex',alignItems:'center',gap:7,padding:'7px 8px',borderRadius:7,
    background:active?cfg.bg:'transparent',
    border:`1.5px solid ${active?cfg.color+'40':'transparent'}`,
    cursor:'pointer',transition:'all 0.12s',width:'100%',
  }}
    onMouseEnter={e=>!active&&(e.currentTarget.style.background=C.hover)}
    onMouseLeave={e=>!active&&(e.currentTarget.style.background='transparent')}>
    <Ico n={cfg.icon} s={12} c={active?cfg.color:C.textMid}/>
    <span style={{fontSize:12,fontWeight:active?700:500,color:active?cfg.color:C.textMid}}>
      {cfg.label}
    </span>
    {active&&<div style={{marginLeft:'auto',width:5,height:5,borderRadius:'50%',background:cfg.color}}/>}
  </button>
);

const FeatureList = ({agent}) => {
  const items = agent
    ?[{n:'brain',  c:C.violet,t:'Query Analysis',  d:'Classifies question type'},
      {n:'zap',    c:C.blue,  t:'Smart Selection',  d:'Picks optimal strategy'},
      {n:'chart',  c:C.green, t:'Quality Check',    d:'Scores & refines answers'},
      {n:'repeat', c:C.cyan,  t:'Self-Improving',   d:'Iterates to confidence'}]
    :[{n:'search', c:C.blue,  t:'Semantic Search',  d:'Vector embeddings'},
      {n:'layers', c:C.green, t:'Keyword Search',   d:'BM25 term ranking'},
      {n:'zap',    c:C.violet,t:'Hybrid Fusion',    d:'Reciprocal rank fusion'},
      {n:'sparkle',c:C.cyan,  t:'CLIP Retrieval',   d:'True multimodal RAG'}];
  return (
    <div style={{display:'flex',flexDirection:'column',gap:6}}>
      {items.map(({n,c,t,d})=>(
        <div key={t} style={{display:'flex',gap:8,alignItems:'flex-start',padding:'3px 0'}}>
          <div style={{width:22,height:22,borderRadius:6,flexShrink:0,
            background:C.hover,border:`1px solid ${C.border}`,
            display:'flex',alignItems:'center',justifyContent:'center'}}>
            <Ico n={n} s={11} c={c}/>
          </div>
          <div>
            <div style={{fontSize:11.5,fontWeight:600,color:C.text}}>{t}</div>
            <div style={{fontSize:10.5,color:C.textLow,marginTop:1}}>{d}</div>
          </div>
        </div>
      ))}
    </div>
  );
};

const SBtn = ({icon,label,onClick,danger}) => (
  <button onClick={onClick} style={{
    width:'100%',display:'flex',alignItems:'center',gap:7,
    padding:'7px 8px',borderRadius:7,border:'none',cursor:'pointer',
    background:'transparent',color:danger?C.red:C.textMid,
    fontSize:11.5,fontWeight:500,fontFamily:"'Geist',sans-serif",
    transition:'background 0.12s',marginBottom:1,
  }}
    onMouseEnter={e=>e.currentTarget.style.background=danger?C.redLight:C.hover}
    onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
    <Ico n={icon} s={12} c={danger?C.red:C.textMid}/>{label}
  </button>
);

/* ════════════════════════════════════════════════════════════════════════════
   CHAT
════════════════════════════════════════════════════════════════════════════ */
const Welcome = ({agent}) => (
  <div style={{flex:1,display:'flex',alignItems:'center',justifyContent:'center'}}>
    <div style={{textAlign:'center',maxWidth:460,animation:'fadeUp 0.5s both'}}>
      <div style={{
        width:56,height:56,borderRadius:14,margin:'0 auto 16px',
        background:'linear-gradient(135deg,#dbeafe,#ede9fe)',
        border:`1px solid ${C.border}`,
        display:'flex',alignItems:'center',justifyContent:'center',
      }}>
        <Ico n="layers" s={24} c={C.blue}/>
      </div>
      <h2 style={{fontSize:20,fontWeight:700,color:C.text,marginBottom:6,letterSpacing:'-0.02em'}}>
        DocuMind AI
      </h2>
      <p style={{fontSize:13,color:C.textMid,lineHeight:1.7,marginBottom:22}}>
        {agent
          ?'Upload a PDF and let the agent intelligently search, evaluate, and synthesize answers.'
          :'True multimodal RAG with vision understanding, hybrid retrieval, and LLM-powered evaluation.'}
      </p>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:8}}>
        {[
          {n:'sparkle',c:C.violet,t:'Vision AI',     d:'Understands charts & diagrams'},
          {n:'zap',    c:C.blue,  t:'Hybrid Search', d:'Semantic + keyword + CLIP'},
          {n:'chart',  c:C.green, t:'LLM Judge',     d:'Faithfulness scoring'},
          {n:'globe',  c:C.cyan,  t:'Factual Check', d:'External knowledge verification'},
        ].map(({n,c,t,d})=>(
          <div key={t} style={{padding:'10px 12px',borderRadius:8,
            background:C.surface,border:`1px solid ${C.border}`,textAlign:'left'}}>
            <div style={{display:'flex',alignItems:'center',gap:5,marginBottom:2}}>
              <Ico n={n} s={12} c={c}/>
              <span style={{fontSize:11.5,fontWeight:700,color:C.text}}>{t}</span>
            </div>
            <div style={{fontSize:10.5,color:C.textLow,lineHeight:1.5}}>{d}</div>
          </div>
        ))}
      </div>
    </div>
  </div>
);

const Typing = () => (
  <div style={{display:'flex',animation:'fadeUp 0.25s both'}}>
    <div style={{
      padding:'10px 14px',borderRadius:'4px 12px 12px 12px',
      background:C.surface,border:`1px solid ${C.border}`,
      boxShadow:'0 1px 3px rgba(0,0,0,0.06)',
      display:'flex',gap:4,alignItems:'center',
    }}>
      {[0,0.18,0.36].map((d,i)=>(
        <div key={i} style={{width:6,height:6,borderRadius:'50%',background:C.blue,
          animation:`pulseDot 1.3s infinite ease-in-out`,animationDelay:`${d}s`}}/>
      ))}
    </div>
  </div>
);

const Bubble = ({m,i,open,toggle,onSrc}) => {
  if(m.t==='user') return (
    <div className="fade-up" style={{display:'flex',justifyContent:'flex-end'}}>
      <div style={{
        maxWidth:'72%',padding:'10px 15px',
        borderRadius:'14px 4px 14px 14px',
        background:'linear-gradient(135deg,#1d4ed8,#7c3aed)',
        color:'#fff',fontSize:13.5,lineHeight:1.65,
        boxShadow:'0 2px 8px rgba(37,99,235,0.2)',
      }}>{m.text}</div>
    </div>
  );

  if(m.t==='err') return (
    <div className="fade-up" style={{display:'flex'}}>
      <div style={{
        maxWidth:'72%',padding:'9px 13px',
        borderRadius:'4px 12px 12px 12px',
        background:C.redLight,border:`1px solid #fca5a5`,
        color:C.red,fontSize:12.5,display:'flex',gap:7,alignItems:'flex-start',
      }}>
        <Ico n="warn" s={13} c={C.red} st={{flexShrink:0,marginTop:2}}/>{m.text}
      </div>
    </div>
  );

  if(m.t==='sys') return (
    <div className="fade-up" style={{display:'flex',justifyContent:'center'}}>
      <div style={{
        padding:'5px 12px',borderRadius:99,
        background:C.surface,border:`1px solid ${C.border}`,
        fontSize:11.5,color:C.textMid,
        display:'flex',alignItems:'center',gap:7,flexWrap:'wrap',
        justifyContent:'center',boxShadow:'0 1px 2px rgba(0,0,0,0.04)',
      }}>
        <Ico n="check" s={11} c={C.green}/>
        {m.text}
        {m.up&&<>
          <Tag color={C.textMid} bg={C.hover} border={C.border}>{m.up.text} text</Tag>
          {m.up.vis&&<Tag color={C.violet} bg={C.violetLight} border="#ddd6fe">{m.up.ocr} vision</Tag>}
          {m.up.tables>0&&<Tag color={C.cyan} bg={C.cyanLight} border="#a5f3fc">{m.up.tables} tables</Tag>}
          <Tag color={C.green} bg={C.greenLight} border="#bbf7d0">{m.up.total} total</Tag>
        </>}
      </div>
    </div>
  );

  /* assistant */
  const cfg = m.sm ? M[m.sm] : null;
  return (
    <div className="fade-up" style={{display:'flex'}}>
      <div style={{maxWidth:'82%'}}>
        <div style={{
          padding:'13px 16px',borderRadius:'4px 14px 14px 14px',
          background:C.surface,border:`1px solid ${C.border}`,
          color:C.text,fontSize:13.5,lineHeight:1.75,
          boxShadow:'0 1px 3px rgba(0,0,0,0.05)',
        }}>{m.text}</div>

        {/* meta row */}
        <div style={{display:'flex',flexWrap:'wrap',gap:5,marginTop:6,paddingLeft:2}}>
          {m.isAgent&&<>
            <Tag color={C.violet} bg={C.violetLight} border="#ddd6fe">🤖 Agent</Tag>
            {m.conf!=null&&<Tag color={C.green} bg={C.greenLight} border="#bbf7d0">{(m.conf*100).toFixed(0)}% conf</Tag>}
            {m.qs!=null&&<Tag color={C.blue} bg={C.blueLight} border={C.blueBorder}>Q {m.qs.toFixed(2)}</Tag>}
            {m.meta?.iterations&&<Tag color={C.amber} bg={C.amberLight} border="#fde68a">{m.meta.iterations} iter</Tag>}
            {m.meta?.total_time!=null&&<Tag color={C.textMid} bg={C.hover} border={C.border}>{msf(m.meta.total_time)}</Tag>}
            <button onClick={()=>toggle(i)} style={{fontSize:11,color:C.blue,background:'none',border:'none',
              cursor:'pointer',display:'flex',alignItems:'center',gap:3,fontFamily:"'Geist',sans-serif",
              padding:'2px 4px'}}>
              <Ico n={open[i]?'up':'down'} s={10} c={C.blue}/>
              {open[i]?'Hide':'Show'} reasoning
            </button>
          </>}
          {!m.isAgent&&cfg&&<>
            <Tag color={cfg.color} bg={cfg.bg}>{cfg.label}</Tag>
            {m.st!=null&&<Tag color={C.textMid} bg={C.hover} border={C.border}>{msf(m.st)}</Tag>}
            {m.ctx!=null&&<Tag color={C.textMid} bg={C.hover} border={C.border}>{m.ctx} sources</Tag>}
            {m.clip&&<Tag color={C.violet} bg={C.violetLight} border="#ddd6fe">🔮 CLIP</Tag>}
          </>}
          {m.srcs?.length>0&&(
            <button onClick={()=>onSrc(m.srcs)} style={{fontSize:11,color:C.blue,background:'none',
              border:'none',cursor:'pointer',display:'flex',alignItems:'center',gap:3,
              fontFamily:"'Geist',sans-serif",padding:'2px 4px'}}>
              <Ico n="eye" s={10} c={C.blue}/>{m.srcs.length} sources
            </button>
          )}
        </div>

        {/* agent reasoning */}
        {m.isAgent&&open[i]&&m.meta&&(
          <div style={{
            marginTop:7,padding:'10px 13px',borderRadius:9,
            background:C.violetLight,border:`1px solid #ddd6fe`,
            animation:'fadeUp 0.2s both',
          }}>
            <div style={{fontSize:10.5,fontWeight:700,color:C.violet,marginBottom:6,
              display:'flex',gap:5,alignItems:'center'}}>
              <Ico n="brain" s={10} c={C.violet}/> DECISION PROCESS
            </div>
            {m.meta.agent_thoughts?.map((t,j)=>(
              <div key={j} style={{display:'flex',gap:7,marginBottom:3}}>
                <span style={{fontSize:9.5,color:C.textLow,fontFamily:"'Geist Mono',monospace",
                  flexShrink:0,marginTop:1}}>{String(j+1).padStart(2,'0')}</span>
                <span style={{fontSize:11.5,color:C.textMid,lineHeight:1.55}}>{t}</span>
              </div>
            ))}
            {m.meta.workflow_path?.length>0&&(
              <div style={{marginTop:7,paddingTop:7,borderTop:`1px solid #ddd6fe`,
                display:'flex',gap:4,flexWrap:'wrap',alignItems:'center'}}>
                {m.meta.workflow_path.map((s,j)=>(
                  <React.Fragment key={s}>
                    <span style={{fontSize:9.5,color:C.violet,fontFamily:"'Geist Mono',monospace",
                      padding:'1px 5px',borderRadius:4,background:'#fff',border:`1px solid #ddd6fe`}}>{s}</span>
                    {j<m.meta.workflow_path.length-1&&<span style={{color:C.textXLow,fontSize:9}}>→</span>}
                  </React.Fragment>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

const AChip = ({icon,label,onClick,disabled,loading:ld,color=C.textMid}) => (
  <button onClick={onClick} disabled={disabled} style={{
    display:'flex',alignItems:'center',gap:5,padding:'5px 10px',borderRadius:6,
    background:disabled?'transparent':C.surface,
    border:`1px solid ${disabled?C.border:C.borderMid}`,
    color:disabled?C.textXLow:color,
    fontSize:11.5,fontWeight:500,cursor:disabled?'not-allowed':'pointer',
    fontFamily:"'Geist',sans-serif",transition:'all 0.12s',
  }}
    onMouseEnter={e=>!disabled&&(e.currentTarget.style.borderColor=color)}
    onMouseLeave={e=>!disabled&&(e.currentTarget.style.borderColor=C.borderMid)}>
    {ld?<Spin s={11} c={color}/>:<Ico n={icon} s={11} c={disabled?C.textXLow:color}/>}
    {label}
  </button>
);

/* ════════════════════════════════════════════════════════════════════════════
   MODAL
════════════════════════════════════════════════════════════════════════════ */
const MWrap = ({title,icon,subtitle,children,onClose,wide=false}) => (
  <div onClick={onClose} style={{
    position:'fixed',inset:0,background:'rgba(28,25,23,0.45)',backdropFilter:'blur(4px)',
    display:'flex',alignItems:'center',justifyContent:'center',zIndex:1000,padding:20,
  }}>
    <div onClick={e=>e.stopPropagation()} style={{
      width:'100%',maxWidth:wide?920:600,maxHeight:'88vh',
      display:'flex',flexDirection:'column',
      background:C.surface,borderRadius:14,border:`1px solid ${C.border}`,
      boxShadow:'0 20px 60px rgba(28,25,23,0.2)',overflow:'hidden',
      animation:'fadeUp 0.22s cubic-bezier(.16,1,.3,1) both',
    }}>
      <div style={{
        display:'flex',alignItems:'center',justifyContent:'space-between',
        padding:'13px 18px',borderBottom:`1px solid ${C.border}`,flexShrink:0,
        background:C.surfaceAlt,
      }}>
        <div style={{display:'flex',alignItems:'center',gap:9}}>
          <div style={{width:28,height:28,borderRadius:7,background:C.blueLight,
            border:`1px solid ${C.blueBorder}`,
            display:'flex',alignItems:'center',justifyContent:'center'}}>
            <Ico n={icon} s={13} c={C.blue}/>
          </div>
          <div>
            <div style={{fontSize:13.5,fontWeight:700,color:C.text}}>{title}</div>
            {subtitle&&<div style={{fontSize:10.5,color:C.textLow,marginTop:1,
              maxWidth:440,overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>{subtitle}</div>}
          </div>
        </div>
        <button onClick={onClose} style={{width:26,height:26,borderRadius:6,border:`1px solid ${C.border}`,
          background:C.surface,cursor:'pointer',display:'flex',alignItems:'center',justifyContent:'center'}}>
          <Ico n="x" s={12} c={C.textMid}/>
        </button>
      </div>
      <div style={{flex:1,overflowY:'auto',padding:18}}>{children}</div>
    </div>
  </div>
);

/* ════════════════════════════════════════════════════════════════════════════
   MODAL BODIES
════════════════════════════════════════════════════════════════════════════ */
const SrcsBody = ({srcs}) => (
  <div style={{display:'flex',flexDirection:'column',gap:10}}>
    {srcs.map((s,i)=>{
      const cfg=SRC[s.metadata?.type]||SRC.text;
      const sc=s.score||0.5;
      const scoreColor=sc>0.7?C.green:sc>0.4?C.amber:C.red;
      return(
        <div key={i} style={{borderRadius:9,border:`1px solid ${C.border}`,overflow:'hidden'}}>
          <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',
            padding:'7px 12px',borderBottom:`1px solid ${C.border}`,background:C.surfaceAlt}}>
            <div style={{display:'flex',alignItems:'center',gap:6}}>
              <span style={{fontSize:10,color:C.textLow,fontFamily:"'Geist Mono',monospace"}}>
                #{String(i+1).padStart(2,'0')}
              </span>
              <Tag color={cfg.color} bg={cfg.bg}>
                <Ico n={cfg.icon} s={9} c={cfg.color}/> {cfg.label}
              </Tag>
            </div>
            <span style={{fontSize:10,color:C.textLow,fontFamily:"'Geist Mono',monospace"}}>
              {s.metadata?.source} · p.{s.metadata?.page}
            </span>
          </div>
          <div style={{padding:12,fontSize:12,color:C.textMid,lineHeight:1.7,
            maxHeight:130,overflowY:'auto',background:C.surface}}>{s.text}</div>
          <div style={{padding:'6px 12px',borderTop:`1px solid ${C.border}`,
            display:'flex',alignItems:'center',gap:8,background:C.surfaceAlt}}>
            <div style={{flex:1}}><Bar v={sc} color={scoreColor} h={3}/></div>
            <span style={{fontSize:10,color:C.textLow,fontFamily:"'Geist Mono',monospace",flexShrink:0}}>
              {(sc*100).toFixed(0)}%
            </span>
          </div>
        </div>
      );
    })}
  </div>
);

const CompBody = ({data}) => (
  <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:12}}>
    {['vector','bm25','hybrid'].map(k=>{
      const d=data[k]; const cfg=M[k];
      return(
        <div key={k} style={{borderRadius:9,border:`1px solid ${cfg.color}30`,
          background:cfg.bg,overflow:'hidden'}}>
          <div style={{padding:'8px 12px',borderBottom:`1px solid ${cfg.color}25`,
            display:'flex',alignItems:'center',gap:6}}>
            <Ico n={cfg.icon} s={12} c={cfg.color}/>
            <span style={{fontSize:12,fontWeight:700,color:cfg.color}}>{cfg.label}</span>
            <span style={{marginLeft:'auto',fontSize:10,color:C.textMid,fontFamily:"'Geist Mono',monospace"}}>
              {msf(d.search_time)}
            </span>
          </div>
          <div style={{padding:12,background:C.surface}}>
            <div style={{display:'flex',justifyContent:'space-between',fontSize:10.5,marginBottom:7}}>
              <span style={{color:C.textLow}}>Sources</span>
              <span style={{color:C.textMid,fontFamily:"'Geist Mono',monospace"}}>{d.sources?.length??0}</span>
            </div>
            <div style={{fontSize:12,color:C.textMid,lineHeight:1.7,maxHeight:180,overflowY:'auto'}}>{d.answer}</div>
          </div>
        </div>
      );
    })}
  </div>
);

const EvalBody = ({data}) => {
  const w=data.winner?.overall;
  const fc=data.factual_check;

  const verdictCfg = v => ({
    correct:           {color:C.green,  bg:C.greenLight,  border:'#bbf7d0', label:'CORRECT'},
    partially_correct: {color:C.amber,  bg:C.amberLight,  border:'#fde68a', label:'PARTIAL'},
    incorrect:         {color:C.red,    bg:C.redLight,    border:'#fca5a5', label:'INCORRECT'},
    unverifiable:      {color:C.textMid,bg:C.hover,       border:C.border,  label:'UNVERIFIABLE'},
  }[v]||{color:C.textMid,bg:C.hover,border:C.border,label:'UNVERIFIABLE'});

  return(
    <div style={{display:'flex',flexDirection:'column',gap:12}}>

      {/* Winner */}
      {w&&(
        <div style={{padding:'12px 14px',borderRadius:10,
          background:C.blueLight,border:`1px solid ${C.blueBorder}`}}>
          <div style={{display:'flex',alignItems:'center',gap:9,marginBottom:9}}>
            <div style={{width:28,height:28,borderRadius:7,background:'#fff',
              border:`1px solid ${C.blueBorder}`,
              display:'flex',alignItems:'center',justifyContent:'center'}}>
              <Ico n="trophy" s={13} c={C.blue}/>
            </div>
            <div>
              <div style={{fontSize:13,fontWeight:700,color:C.blue}}>
                {w.method?.toUpperCase()} wins · {fmt3(w.score)}
              </div>
              <div style={{display:'flex',gap:10,marginTop:2}}>
                {['fastest','most_relevant'].map(k=>data.winner[k]&&(
                  <span key={k} style={{fontSize:10.5,color:C.textMid}}>
                    {k==='fastest'?'⚡':'🎯'} {data.winner[k].method?.toUpperCase()}
                    {k==='fastest'?` ${msf(data.winner[k].time)}`
                      :` ${fmt3(data.winner[k].score)}`}
                  </span>
                ))}
              </div>
            </div>
          </div>
          {w.reasoning&&(
            <div style={{padding:'9px 11px',borderRadius:7,background:'rgba(255,255,255,0.7)',
              border:`1px solid ${C.blueBorder}`,fontSize:12,color:C.textMid,lineHeight:1.7,
              display:'flex',gap:6,alignItems:'flex-start'}}>
              <Ico n="info" s={12} c={C.blue} st={{flexShrink:0,marginTop:2}}/>
              <span>{w.reasoning.split('**').map((p,i)=>
                i%2===1?<strong key={i} style={{color:C.text,fontWeight:700}}>{p}</strong>:p
              )}</span>
            </div>
          )}
        </div>
      )}

      {/* Factual check */}
      {fc&&(()=>{
        const vc=verdictCfg(fc.verdict);
        return(
          <div style={{padding:'11px 13px',borderRadius:10,
            background:vc.bg,border:`1px solid ${vc.border}`}}>
            <div style={{display:'flex',alignItems:'center',justifyContent:'space-between',marginBottom:7}}>
              <div style={{display:'flex',alignItems:'center',gap:7}}>
                <Ico n="globe" s={13} c={vc.color}/>
                <span style={{fontSize:12.5,fontWeight:700,color:C.text}}>External Factual Check</span>
              </div>
              <Tag color={vc.color} bg={vc.bg} border={vc.border}>{vc.label}</Tag>
            </div>
            {fc.factual_accuracy!=null&&(
              <div style={{marginBottom:7}}>
                <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                  <span style={{fontSize:11,color:C.textMid}}>Factual Accuracy</span>
                  <span style={{fontSize:11,color:vc.color,fontFamily:"'Geist Mono',monospace",fontWeight:700}}>
                    {pct(fc.factual_accuracy)}
                  </span>
                </div>
                <Bar v={fc.factual_accuracy} color={vc.color}/>
              </div>
            )}
            {fc.external_context&&(
              <div style={{fontSize:11.5,color:C.textMid,fontStyle:'italic',lineHeight:1.6,
                padding:'7px 9px',borderRadius:6,background:'rgba(255,255,255,0.6)',
                border:`1px solid ${vc.border}`,
                marginBottom:fc.issues?.length>0?7:0}}>
                📚 {fc.external_context}
              </div>
            )}
            {fc.issues?.map((is,i)=>(
              <div key={i} style={{fontSize:11,color:C.red,display:'flex',gap:5,marginTop:3}}>
                <Ico n="warn" s={11} c={C.red} st={{flexShrink:0,marginTop:1}}/>{is}
              </div>
            ))}
            {!fc.verifiable&&(
              <div style={{fontSize:10.5,color:C.textLow,marginTop:5,fontStyle:'italic'}}>
                ℹ Document-specific — cannot be fully verified against general knowledge.
              </div>
            )}
          </div>
        );
      })()}

      {/* Method cards */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:10}}>
        {['vector','bm25','hybrid'].map(k=>{
          const d=data.methods?.[k]; const cfg=M[k];
          if(!d?.success) return(
            <div key={k} style={{borderRadius:9,border:`1px solid ${C.border}`,
              background:C.surfaceAlt,padding:12,textAlign:'center'}}>
              <div style={{fontSize:11,color:C.red}}>{k.toUpperCase()} failed</div>
            </div>
          );
          const met=d.metrics; const lj=d.llm_judge;
          return(
            <div key={k} style={{borderRadius:10,border:`1px solid ${cfg.color}25`,
              overflow:'hidden',background:C.surface}}>
              <div style={{padding:'8px 11px',borderBottom:`1px solid ${cfg.color}20`,
                display:'flex',alignItems:'center',gap:6,background:cfg.bg}}>
                <Ico n={cfg.icon} s={12} c={cfg.color}/>
                <span style={{fontSize:11.5,fontWeight:700,color:cfg.color}}>{cfg.label}</span>
                {d.is_extractive&&(
                  <span style={{marginLeft:'auto',fontSize:9,color:C.amber,
                    background:C.amberLight,border:'1px solid #fde68a',
                    padding:'1px 5px',borderRadius:4,fontWeight:700}}>EXTRACTIVE</span>
                )}
              </div>
              <div style={{padding:11,display:'flex',flexDirection:'column',gap:6}}>
                <div style={{display:'flex',justifyContent:'space-between',fontSize:10.5}}>
                  <span style={{color:C.textLow}}>Time</span>
                  <span style={{color:C.textMid,fontFamily:"'Geist Mono',monospace"}}>{msf(d.response_time)}</span>
                  <span style={{color:C.textLow}}>Sources</span>
                  <span style={{color:C.textMid,fontFamily:"'Geist Mono',monospace"}}>{d.sources_count}</span>
                </div>
                <ME label="Answer Rel."  v={met.answer_relevance}  c={C.blue}  />
                <ME label="Precision"    v={met.context_precision} c={C.green} />
                <ME label="Faithfulness" v={met.faithfulness}      c={C.cyan}  />
                {lj?.faithfulness!=null&&<>
                  <Divider m="2px 0"/>
                  <div style={{fontSize:9.5,color:C.violet,fontWeight:700,
                    display:'flex',gap:4,alignItems:'center',marginBottom:1}}>
                    <Ico n="brain" s={9} c={C.violet}/> LLM JUDGE
                  </div>
                  <ME label="Faithfulness" v={lj.faithfulness} c={C.violet}/>
                  <ME label="Completeness" v={lj.completeness} c="#6d28d9"/>
                  {lj.reasoning&&<div style={{fontSize:10,color:C.textLow,fontStyle:'italic',
                    lineHeight:1.5,marginTop:2}}>"{lj.reasoning}"</div>}
                </>}
                <Divider m="2px 0"/>
                <div>
                  <div style={{display:'flex',justifyContent:'space-between',marginBottom:3}}>
                    <span style={{fontSize:11,fontWeight:700,color:C.text}}>Overall</span>
                    <span style={{fontSize:13,fontWeight:700,color:cfg.color,
                      fontFamily:"'Geist Mono',monospace"}}>{pct(met.overall_score)}</span>
                  </div>
                  <Bar v={met.overall_score} color={cfg.color} h={4}/>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

const ConsBody = ({data}) => {
  const ok=data.is_consistent;
  return(
    <div style={{display:'flex',flexDirection:'column',gap:11}}>
      <div style={{padding:'12px 14px',borderRadius:10,
        background:ok?C.greenLight:C.amberLight,
        border:`1px solid ${ok?'#bbf7d0':'#fde68a'}`,
        display:'flex',alignItems:'center',gap:10}}>
        <div style={{width:32,height:32,borderRadius:8,
          background:'rgba(255,255,255,0.6)',border:`1px solid ${ok?'#bbf7d0':'#fde68a'}`,
          display:'flex',alignItems:'center',justifyContent:'center'}}>
          <Ico n={ok?'check':'warn'} s={15} c={ok?C.green:C.amber}/>
        </div>
        <div>
          <div style={{fontSize:13.5,fontWeight:700,color:ok?C.green:C.amber}}>
            {ok?'Consistent':'Inconsistent'}
          </div>
          <div style={{fontSize:11.5,color:C.textMid,marginTop:2}}>
            Score {fmt3(data.consistency_score)} · {data.n_runs} runs · {data.mode?.toUpperCase()} mode
          </div>
        </div>
      </div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(2,1fr)',gap:8}}>
        {[['Mean Score',fmt3(data.mean_score),C.blue],
          ['Std Dev',fmt3(data.std_score),data.std_score<0.05?C.green:C.amber],
          ['Answer Sim.',fmt3(data.answer_similarity),C.cyan],
          ['Consistency',fmt3(data.consistency_score),ok?C.green:C.amber]]
          .map(([l,v,c])=>(
          <div key={l} style={{padding:'9px 11px',borderRadius:8,
            background:C.surface,border:`1px solid ${C.border}`}}>
            <div style={{fontSize:10.5,color:C.textLow,marginBottom:4}}>{l}</div>
            <div style={{fontSize:16,fontWeight:700,color:c,
              fontFamily:"'Geist Mono',monospace"}}>{v}</div>
          </div>
        ))}
      </div>
      {data.scores?.length>0&&(
        <div>
          <div style={{fontSize:10,color:C.textLow,marginBottom:7,
            fontWeight:700,textTransform:'uppercase',letterSpacing:'0.06em'}}>Per-run scores</div>
          <div style={{display:'flex',gap:8}}>
            {data.scores.map((s,i)=>{
              const c=s>0.6?C.green:s>0.4?C.amber:C.red;
              return(
                <div key={i} style={{flex:1,padding:'9px 11px',borderRadius:8,
                  background:C.surface,border:`1px solid ${C.border}`,textAlign:'center'}}>
                  <div style={{fontSize:10,color:C.textLow,marginBottom:5}}>Run {i+1}</div>
                  <div style={{fontSize:15,fontWeight:700,marginBottom:7,
                    fontFamily:"'Geist Mono',monospace",color:c}}>{fmt3(s)}</div>
                  <Bar v={s} color={c} h={3}/>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

const ME = ({label,v,c}) => (
  <div>
    <div style={{display:'flex',justifyContent:'space-between',marginBottom:2}}>
      <span style={{fontSize:10,color:C.textLow}}>{label}</span>
      <span style={{fontSize:10,color:c,fontFamily:"'Geist Mono',monospace",fontWeight:700}}>{pct(v)}</span>
    </div>
    <Bar v={v} color={c} h={3}/>
  </div>
);