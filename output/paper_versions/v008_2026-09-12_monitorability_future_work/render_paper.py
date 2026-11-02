"""Render the evidence-bounded Markdown research manuscript as a PDF draft."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, CondPageBreak
from pypdf import PdfReader


def math_markup(text):
    text=text.replace(r'\alpha','α').replace(r'\Delta','Δ')
    text=re.sub(r'\{\\mathrm\{([^}]+)\}\}',r'\1',text)
    text=re.sub(r'_\{([^}]+)\}',r'<sub>\1</sub>',text)
    text=re.sub(r'_([A-Za-z0-9]+)',r'<sub>\1</sub>',text)
    if '\\' in text:raise ValueError('Unhandled math notation: '+text)
    return text


def markup(text):
    text=text.replace('F_delta', r'\(F_{\Delta}\)')
    text=text.translate(str.maketrans({'—':'-','–':'-','‑':'-'}))
    fragments=[]
    def protected(value):
        fragments.append(value)
        return f'PLACEHOLDER{len(fragments)-1}END'
    text=re.sub(r'\\\((.*?)\\\)',lambda m:protected(math_markup(m.group(1))),text)
    text=re.sub(r'`([^`]+)`',lambda m:protected('<font name="Courier" size="8.3">'+html.escape(m.group(1))+'</font>'),text)
    def link(match):
        title,url=match.groups()
        if not url.startswith('https://'):raise ValueError('Unexpected citation URL')
        return protected('<link href="'+html.escape(url,quote=True)+'" color="#245b86">'+html.escape(title)+'</link>')
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,text)
    text=html.escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    for index,value in enumerate(fragments):text=text.replace(f'PLACEHOLDER{index}END',value)
    return text


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path(__file__).resolve().with_name('PAPER.md'))
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().with_name('paper.pdf'))
    args=p.parse_args()
    source=args.source.read_text(encoding='utf-8')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name,file in [('Paper','times.ttf'),('Paper-Bold','timesbd.ttf'),('Paper-Italic','timesi.ttf'),('Paper-BoldItalic','timesbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
    pdfmetrics.registerFontFamily('Paper',normal='Paper',bold='Paper-Bold',italic='Paper-Italic',boldItalic='Paper-BoldItalic')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='PaperBody',fontName='Paper',fontSize=10.5,leading=14.2,spaceAfter=7,
        allowWidows=0,allowOrphans=0,splitLongWords=True))
    styles.add(ParagraphStyle(name='PaperTitle',fontName='Paper-Bold',fontSize=21,leading=25,spaceAfter=14,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperSection',fontName='Paper-Bold',fontSize=13.5,leading=17,spaceBefore=11,spaceAfter=7,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperSubsection',fontName='Paper-Bold',fontSize=11.7,leading=15,spaceBefore=8,spaceAfter=6,keepWithNext=True))
    styles.add(ParagraphStyle(name='PaperCell',fontName='Paper',fontSize=9.1,leading=11.6,splitLongWords=True))
    styles.add(ParagraphStyle(name='PaperMath',parent=styles['PaperBody'],leftIndent=16,spaceBefore=5,spaceAfter=10))
    styles.add(ParagraphStyle(name='PaperAppendixBody',parent=styles['PaperBody'],spaceAfter=4))
    story=[];headings=[];table_cells=[]
    in_appendix=False
    lines=source.splitlines();index=0;table_count=0
    while index<len(lines):
        line=lines[index].strip()
        if not line:index+=1;continue
        figure=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if figure:
            caption,relative=figure.groups()
            path=(args.source.parent/relative).resolve()
            if not path.is_relative_to(args.source.parent.resolve()):raise ValueError('Figure outside source tree')
            from PIL import Image as PILImage
            with PILImage.open(path) as bitmap:width,height=bitmap.size
            scale=min(468/width,430/height)
            story.append(KeepTogether([Image(str(path),width=width*scale,height=height*scale),
                Spacer(1,5),Paragraph(markup(caption),styles['PaperCell']),Spacer(1,10)]))
            index+=1;continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));text=line[level:].strip()
            if text.startswith('Appendix '):
                in_appendix=True
                story.append(CondPageBreak(150))
            headings.append(text)
            style='PaperTitle' if level==1 else 'PaperSection' if level==2 else 'PaperSubsection'
            story.append(Paragraph(markup(text),styles[style]));index+=1;continue
        if line==r'\[':
            math=[];index+=1
            while index<len(lines) and lines[index].strip()!=r'\]':math.append(lines[index].strip());index+=1
            story.append(Paragraph(math_markup(' '.join(math)),styles['PaperMath']));index+=1;continue
        if line.startswith('|'):
            rows=[]
            while index<len(lines) and lines[index].strip().startswith('|'):
                cells=[c.strip() for c in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells);table_cells.extend(cells)
                index+=1
            count=len(rows[0])
            if any(len(row)!=count for row in rows):raise ValueError('Inconsistent table column count')
            available=468
            first_fraction=.35 if count>=4 else .48
            widths=[available*first_fraction]+[available*(1-first_fraction)/(count-1)]*(count-1)
            data=[[Paragraph(('<b>'+markup(c)+'</b>') if i==0 else markup(c),styles['PaperCell']) for c in row] for i,row in enumerate(rows)]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=0)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f6')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7b8b9b')),
                ('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#9aa7b2')),
                ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,9)]);table_count+=1;continue
        paragraph=[]
        while index<len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(('#','|')) and lines[index].strip()!=r'\[':
            paragraph.append(lines[index].strip());index+=1
        story.append(Paragraph(markup(' '.join(paragraph)),styles['PaperAppendixBody' if in_appendix else 'PaperBody']))
    def furniture(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor('#c2cbd2'));canvas.setLineWidth(.4)
        canvas.line(72,46,540,46);canvas.setFont('Paper',8.5);canvas.setFillColor(colors.HexColor('#56616d'))
        canvas.drawString(72,34,'Version 8 | Monitorability future work | 12 September 2026')
        canvas.drawRightString(540,34,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(args.output),pagesize=letter,rightMargin=72,leftMargin=72,topMargin=55,bottomMargin=61,
        title=headings[0],author='',subject='Tracking shared parameter effects through transformer passes; cross-family geometry and intervention-change fidelity')
    doc.build(story,onFirstPage=furniture,onLaterPages=furniture)
    reader=PdfReader(args.output);text='\n'.join(page.extract_text() or '' for page in reader.pages)
    normalized=re.sub(r'\s+','',text)
    def plain(value):
        return re.sub(r'\s+','',re.sub(r'<[^>]+>','',html.unescape(markup(value))))
    missing_headings=[h for h in headings if plain(h) not in normalized]
    missing_cells=[c for c in table_cells if plain(c) not in normalized]
    if missing_headings or missing_cells:raise ValueError(dict(missing_headings=missing_headings,missing_cells=missing_cells))
    if '\u25a0' in text or '\ufffd' in text:raise ValueError('Possible missing glyph in extracted PDF')
    report=dict(status='pdf_draft_rendered_text_checks_passed',pages=len(reader.pages),tables=table_count,
        headings_checked=len(headings),table_cells_checked=len(table_cells),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        pdf_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        created_utc=datetime.now(timezone.utc).isoformat(),visual_review_required=True)
    args.output.with_suffix('.receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
