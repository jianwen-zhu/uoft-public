from pathlib import Path
from html import escape
import re

root = Path(__file__).resolve().parent

def section(filename, anchor, heading):
    blocks = re.split(r'\n\s*\n', (root / "content" / filename).read_text().strip())
    output = [f'<section id="{anchor}" aria-labelledby="{anchor}-title">', f'<h2 id="{anchor}-title">{heading}</h2>']
    for block in blocks[1:]:
        text = ' '.join(block.split())
        if re.match(r'Theme \d+:', text):
            text = re.sub(r'^Theme \d+: ', '', text)
            output.append(f'<h3>{escape(text)}</h3>')
        else:
            rendered = escape(text)
            if text.startswith('YTL (Yang Tensor Language)'):
                rendered = rendered.replace('YTL', '<a href="references/ytl-lrm-2.0.html">YTL</a>', 1)
            output.append(f'<p>{rendered}</p>')
    output.append('</section>')
    return '\n'.join(output)

html = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Active research projects with Jianwen Zhu: persistent AI foundations, agentic electronic design automation, and new ideas in combinatorial optimization.">
<title>Active Research · Jianwen Zhu</title>
<link rel="stylesheet" href="style.css">
</head>
<body id="top">
<a class="skip" href="#main">Skip to content</a>
<div class="stationary-header">
<nav aria-label="Research themes">

<div><a href="#foundations">Persistent AI Foundations</a><a href="#eda">Agentic EDA</a><a href="#emerging">Representational Combinatorics</a></div>
</nav>
<header class="hero">
<h1>Jianwen Zhu <span>· Active Research</span></h1>
<p class="affiliation">Electrical &amp; Computer Engineering · University of Toronto</p>
<p class="recruiting">I am looking for talented graduate students to pursue the following active projects.</p>
<p class="student-scope">Students pursue focused projects within these broader research themes.</p>
</header>
</div>
<main id="main">
'''
html += section('agent-persistent.org', 'foundations', 'Persistent AI Foundations')
html += section('agent-eda.org', 'eda', 'Agentic EDA')
html += '''
<section id="emerging" aria-labelledby="emerging-title">
<h2 id="emerging-title">Representational Combinatorics</h2>
<p>How can a better representation make a combinatorial problem easier to solve? We investigate symbolic and learned representations that expose shared structure and enable efficient computation.</p>
<p>Our work on <a href="https://www.eecg.utoronto.ca/~jzhu/publications/symbolic_conf.pdf">symbolic pointer analysis</a> and <a href="https://www.eecg.toronto.edu/~jzhu/publications/pldi04_conf.pdf">symbolic transfer functions</a> demonstrates this approach using binary decision diagrams. Another direction is <a href="https://arxiv.org/abs/1611.09940">neural combinatorial optimization</a>, motivated by the success of learned representations in deep learning. We study latent spaces for search and the recovery of feasible discrete solutions.</p>
<p>We target classic hard combinatorial problems in general and long-standing EDA problems in particular, evaluating solution quality, correctness, and computational cost.</p>
</section>
<footer><p><a href="../past-research/">Selected Work</a></p></footer>
</main>
<script>
const header = document.querySelector('.stationary-header');
new ResizeObserver(() => {
  document.documentElement.style.setProperty('--header-height', `${header.offsetHeight}px`);
}).observe(header);
</script>
</body>
</html>
'''
(root / 'index.html').write_text(html)
print('Generated research-page/index.html from the two source statements.')
