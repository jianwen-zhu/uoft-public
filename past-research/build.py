import json
from pathlib import Path
from html import escape as e
ROOT=Path(__file__).resolve().parent
T=[
('eda','Electronic Design Automation','Design automation must make a growing space of hardware choices tractable. These works connect executable specifications to implementation, and reformulate synthesis and placement so that better designs remain practical to compute.'),
('analysis','Representational Combinatorics','The cost of a combinatorial problem depends on how its possibilities are represented. By encoding sets and relations in decision diagrams, these works exploit shared structure to analyze programs and explore circuit choices without enumerating every case.'),
('performance','Performance Analysis','A useful performance model explains how workload behavior becomes a hardware cost. These works preserve the statistics that matter—signal correlation and uneven write activity—to expose tradeoffs in numerical precision and storage endurance.'),
('computing','Custom Computing','Specialization pays when the organization of hardware matches the structure of the work. These projects examine where data moves, where state lives, and when computation advances—and how to preserve a custom design as its implementation technology changes.'),
('coding','Channel Modeling and Error Control Coding','')]
# Citation counts read directly from Google Scholar on 2026-11-18.
# Curated selection, not an exhaustive author-profile ranking.
P=[
('coding','A Black-box Model of Memory Life-span for NAND Flash Devices','Shu-Yi Jack Wong','PhD dissertation · University of Toronto',2022,None,'https://utoronto.scholaris.ca/server/api/core/bitstreams/2a2ffb1c-73e8-45cd-b646-e4d2e980cc85/content',None),
('coding','FPGA Micro-Architecture-Code Co-Design For Low-Density Parity-Check Codes For Flash Memories','Reza Nakhjavani','PhD dissertation · University of Toronto',2021,None,'https://utoronto.scholaris.ca/server/api/core/bitstreams/2e78113f-d374-40d7-8030-14c6d2347327/content',None),
('eda','Register Pressure Aware Scheduling for High Level Synthesis','R. Beidas, W. S. Mong, J. Zhu','ASP-DAC',2011,7,'https://doi.org/10.1109/ASPDAC.2011.5722234','11014955788268353968'),
('analysis','BddCut: Towards Scalable Symbolic Cut Enumeration','A. C. Ling, J. Zhu, S. D. Brown','ASP-DAC',2007,8,'https://www.eecg.toronto.edu/~brown/papers/aspdac07-ling.pdf','17809966162877568785'),
('computing','VariPipe: Low-overhead Variable-clock Synchronous Pipelines','N. Toosizadeh, S. G. Zaky, J. Zhu','ICCD',2009,9,'https://www.eecg.toronto.edu/~jzhu/publications/navid-veripipe-iccd09.pdf','8143881292227795055'),
('computing','Calligrapher: a new layout-migration engine for hard intellectual property libraries','J. Zhu, F. Fang, Q. Tang','IEEE TCAD',2005,34,'https://www.eecg.toronto.edu/~jzhu/publications/calligrapher_jour.pdf','6390204753888710124'),
('eda','FBDD: A Folded Logic Synthesis System','D. Wu, J. Zhu','ASICON',2005,23,'https://www.eecg.toronto.edu/~jzhu/publications/fbdd_conf.pdf','4657747767116300426'),
('eda','SpecC: Specification Language and Methodology','D. D. Gajski, J. Zhu, R. Dömer, A. Gerstlauer, S. Zhao','Springer',2000,913,'https://link.springer.com/book/10.1007/978-1-4615-4515-6','7347113783153034184'),
('eda','A retargetable, ultra-fast instruction set simulator','J. Zhu, D. D. Gajski','DATE',1999,121,'https://www.eecg.toronto.edu/~jzhu/publications/sim.pdf','11834267141924297164'),
('analysis','Symbolic pointer analysis revisited','J. Zhu, S. Calman','PLDI',2004,133,'https://www.eecg.toronto.edu/~jzhu/publications/pldi04_conf.pdf','3038074044464982255'),
('analysis','Symbolic pointer analysis','J. Zhu','ICCAD',2002,52,'https://www.eecg.toronto.edu/~jzhu/publications/symbolic_conf.pdf','11839314557138386412'),
('analysis','Towards scalable flow and context sensitive pointer analysis','J. Zhu','DAC',2005,51,'https://www.eecg.toronto.edu/~jzhu/publications/qpa_conf.pdf','16510936669755901279'),
('analysis','Scalable synthesis and clustering techniques using decision diagrams','A. C. Ling, J. Zhu, S. D. Brown','IEEE TCAD',2008,11,'https://www.eecg.toronto.edu/~jzhu/publications/TCADBddMap.pdf','15438957944472748250'),
('performance','Write skew and Zipf distribution: Evidence and implications','Y. Yang, J. Zhu','ACM Transactions on Storage',2016,91,'https://dl.acm.org/doi/10.1145/2908557','8181398469109264230'),
('performance','Analytical modeling of garbage collection algorithms in hotness-aware flash-based solid state drives','Y. Yang, J. Zhu','MSST',2014,22,'https://www.eecg.toronto.edu/~jzhu/publications/yang-msst14.pdf','9765305464634945785'),
('eda','Parallelizing simulated annealing-based placement using GPGPU','A. Choong, R. Beidas, J. Zhu','FPL',2010,57,'https://twiki.di.uniroma1.it/pub/CI/WebHome/ParSA-GPGPU.pdf','9801731140054644252'),
('eda','Towards scalable placement for FPGAs','H. Bian, A. C. Ling, A. Choong, J. Zhu','FPGA',2010,56,'https://janders.eecg.utoronto.ca/1387/readings/hannah.pdf','14116613955549681864'),
('computing','A 1 cycle-per-byte XML parsing accelerator','Z. Dai, N. Ni, J. Zhu','FPGA',2010,47,'https://www.eecg.toronto.edu/~jzhu/publications/dai-fpga_2010.pdf','12838141603805746940'),
('computing','Saturating the transceiver bandwidth: Switch fabric design on FPGAs','Z. Dai, J. Zhu','FPGA',2012,37,'https://www.eecg.toronto.edu/~jzhu/publications/dai-fpga_2012.pdf','13930008558713752503'),
('performance','Dynamic-range estimation','B. Wu, J. Zhu, F. N. Najm','IEEE TCAD',2006,34,'https://www.eecg.utoronto.ca/~najm/papers/tcad06-bin.pdf','9530526835281575497')]
thesis_repositories={
 'https://utoronto.scholaris.ca/server/api/core/bitstreams/2a2ffb1c-73e8-45cd-b646-e4d2e980cc85/content':'https://hdl.handle.net/1807/123509',
 'https://utoronto.scholaris.ca/server/api/core/bitstreams/2e78113f-d374-40d7-8030-14c6d2347327/content':'https://hdl.handle.net/1807/106339',
}
records=[dict(zip(['theme','title','authors','venue','year','citations','url','scholar_id'],p)) for p in P]
artifacts={'4657747767116300426': [
 {'label':'Source code · FBDD 1.1 (tar.gz)','url':'https://www.eecg.toronto.edu/~jzhu/releases/fbdd/fbdd-1.1.tar.gz'},
 {'label':'User manual','url':'https://www.eecg.toronto.edu/~jzhu/fbdduser11_web/index.html'},
 {'label':'Project & releases','url':'https://www.eecg.toronto.edu/~jzhu/fbdd_web/index.html'}]}
for record in records:
 if record['url'] in thesis_repositories: record['repository_url']=thesis_repositories[record['url']]
 if record['scholar_id'] in artifacts: record['artifacts']=artifacts[record['scholar_id']]

(ROOT/'publications.json').write_text(json.dumps({'citation_source':'Google Scholar','checked_on':'2026-11-18','selection':'Curated works, grouped by theme; not an exhaustive ranking. Related book chapters and manuals are represented by the SpecC book rather than listed separately.','themes':[dict(zip(['id','title','description'],t)) for t in T],'publications':records},indent=2)+'\n')
def paper(p):
 t,title,authors,venue,year,cites,url,sid=p
 pdf_html=f' · <a class="pdf-link" href="{e(url)}" aria-label="PDF of {e(title)}">PDF</a>' if url.lower().endswith('.pdf') or url in thesis_repositories else ''
 if url in thesis_repositories: pdf_html+=f' · <a href="{e(thesis_repositories[url])}">Repository</a>'
 links=' · '.join(f'<a href="{e(a["url"])}">{e(a["label"])}</a>' for a in artifacts.get(sid,[]))
 artifact_html=f'<p class="artifacts">{links}</p>' if links else ''
 citation_html=f'<a class="citations" href="https://scholar.google.com/scholar?cites={sid}&amp;hl=en" aria-label="{cites} Google Scholar citations for {e(title)}">{cites}</a>' if cites is not None and sid else ''
 return f'<article class="publication" data-citations="{cites if cites is not None else ""}" data-theme="{t}"><div><h3><a href="{e(url)}">{e(title)}</a></h3><p class="authors">{e(authors)}</p><p class="venue">{e(venue)} · {year}{pdf_html}</p>{artifact_html}</div>{citation_html}</article>'
sections=''.join(f'<section id="{tid}" aria-labelledby="{tid}-title"><h2 id="{tid}-title">{e(title)}</h2><div class="papers">'+''.join(paper(p) for p in sorted(P,key=lambda p:-(p[5] if p[5] is not None else -1)) if p[0]==tid)+'</div></section>' for tid,title,desc in T)
nav_labels={'eda':'EDA','analysis':'Combinatorics','performance':'Analysis','computing':'Computing','coding':'Coding'}
nav=''.join(f'<a href="#{tid}">{e(nav_labels[tid])}</a>' for tid,title,_ in T)
html='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Selected Work · Jianwen Zhu</title><meta name="description" content="Selected research and supervision work by Jianwen Zhu, organized by theme."><link rel="stylesheet" href="style.css"></head>
<body id="top">
<a class="skip" href="#main">Skip to content</a>
<div class="stationary-header">
<nav aria-label="Research themes"><div>'''+nav+'''</div></nav>
<header class="hero">
<h1>Jianwen Zhu <span>· Selected Work</span></h1>
<p class="affiliation">Electrical &amp; Computer Engineering · University of Toronto</p>
</header>
</div>
<main id="main">'''+sections+'''
<footer><p><a href="../research-page/index.html">Active Research</a></p></footer></main>
<script>
const header = document.querySelector('.stationary-header');
new ResizeObserver(() => {
  document.documentElement.style.setProperty('--header-height', `${header.offsetHeight}px`);
}).observe(header);
</script>
</body></html>'''
(ROOT/'index.html').write_text(html)
