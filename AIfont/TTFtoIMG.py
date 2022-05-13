from __future__ import print_function, division, absolute_import
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen
from reportlab.graphics.shapes import Path
from reportlab.lib import colors
from reportlab.graphics import renderPM
from reportlab.graphics.shapes import Group, Drawing, scale


class ReportLabPen(BasePen):

    def __init__(self, glyphSet, path=None):
        BasePen.__init__(self, glyphSet)
        if path is None:
            path = Path()
        self.path = path

    def _moveTo(self, p):
        (x, y) = p
        self.path.moveTo(x, y)

    def _lineTo(self, p):
        (x, y) = p
        self.path.lineTo(x, y)

    def _curveToOne(self, p1, p2, p3):
        (x1, y1) = p1
        (x2, y2) = p2
        (x3, y3) = p3
        self.path.curveTo(x1, y1, x2, y2, x3, y3)

    def _closePath(self):
        self.path.closePath()


def ttfToImage(fontName, imagePath, fmt="png"):
    font = TTFont(fontName)
    gs = font.getGlyphSet()
    glyphNames = font.getGlyphNames()
    for i in glyphNames:
        if i[0] == '.':  # 跳过'.notdef', '.null'
            continue
        g = gs[i]
        pen = ReportLabPen(gs, Path(fillColor=colors.red, strokeWidth=5))
        g.draw(pen)
        w, h = g.width, g.width
        g = Group(pen.path)
        g.translate(0, 200)

        d = Drawing(w, h)
        d.add(g)
        imageFile = imagePath + "/" + i + ".png"
        renderPM.drawToFile(d, imageFile, fmt)

        print(i)


ttfToImage(fontName="kaiti.TTF", imagePath="kaiti")