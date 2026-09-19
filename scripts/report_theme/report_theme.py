"""Editorial HTML Report theme v1.0. Copy alongside report.css into a project."""
from html import escape
from pathlib import Path

VERSION = '1.0'
PALETTE = ['#5696b9', '#ce9a48', '#8370b4', '#66717e', '#268a87']


def hero(kicker, title, accent_title, dek, tags):
    return (f'<header><div class="eyebrow">{escape(kicker)}</div>'
            f'<h1>{escape(title)}<br><span>{escape(accent_title)}</span></h1>'
            f'<p class="dek">{escape(dek)}</p><div class="tags">'
            + ''.join(f'<span>{escape(str(t))}</span>' for t in tags) + '</div></header>')


def metric(label, value, note):
    return f'<article class="metric"><span>{escape(label)}</span><strong>{escape(str(value))}</strong><small>{escape(note)}</small></article>'


def finding(index, title, text):
    return f'<article class="finding"><span class="index">{index:02d}</span><h3>{escape(title)}</h3><p>{escape(text)}</p></article>'


def figure_html(chart, caption, note='', data_href=None):
    link=f'<span class="figure-links"><a href="{escape(data_href,quote=True)}" download>Data ↓</a></span>' if data_href else ''
    return f'<figure><figcaption><span>{escape(caption)}</span>{link}</figcaption><div class="chart-scroll">{chart}</div><p class="figure-note">{escape(note)}</p></figure>'


def style_plotly(fig, title, height=580):
    fig.update_layout(title=dict(text=title,font=dict(size=21,family='Arial'),x=.04,y=.97,yanchor='top'),
        height=height,paper_bgcolor='white',plot_bgcolor='#ececf3',
        font=dict(family='Arial',size=12,color='#444'),margin=dict(l=76,r=35,t=120,b=55),
        legend=dict(orientation='h',y=1.025,yanchor='bottom',x=0,font=dict(size=11)),
        hovermode='x unified',colorway=PALETTE)
    fig.update_xaxes(showgrid=True,gridcolor='white',zeroline=False,automargin=True)
    fig.update_yaxes(showgrid=True,gridcolor='white',zerolinecolor='#aaa',automargin=True)
    return fig


def matplotlib_style():
    return {'figure.facecolor':'white','axes.facecolor':'#ececf3','axes.edgecolor':'white',
            'axes.grid':True,'grid.color':'white','grid.linewidth':.8,'font.family':'DejaVu Sans',
            'font.size':10,'text.color':'#282b30','axes.labelcolor':'#636870',
            'xtick.color':'#636870','ytick.color':'#636870','legend.frameon':False,
            'savefig.facecolor':'white','savefig.dpi':160}


def render_page(title, body_html, *, plotly=True, accent='purple'):
    css=Path(__file__).with_name('report.css').read_text(encoding='utf-8')
    if accent=='blue':
        css+='\n:root{--accent:#4c94df}'
    elif accent!='purple':
        raise ValueError('accent must be purple or blue')
    runtime=''
    if plotly:
        from plotly.offline import get_plotlyjs
        runtime='<script>'+get_plotlyjs()+'</script>'
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<link rel="icon" href="data:,"><meta name="report-theme" content="editorial-html-report/'+VERSION+'">'
            f'<title>{escape(title)}</title><style>{css}</style>{runtime}</head>'
            f'<body><main>{body_html}</main></body></html>')
