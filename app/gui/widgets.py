import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget

class Waveform(QWidget):
    def __init__(self):
        super().__init__(); self.setMinimumHeight(105); self.wave=np.zeros(0)
    def set_wave(self,wave):
        self.wave=np.asarray(wave); self.update()
    def paintEvent(self,event):
        painter=QPainter(self); painter.fillRect(self.rect(),QColor('#131b29'))
        painter.setPen(QPen(QColor('#59c5d9'),1))
        if not len(self.wave): painter.drawText(self.rect(),Qt.AlignCenter,'Waveform / 파형'); return
        width=max(1,self.width()); bins=np.array_split(self.wave,min(width,len(self.wave)))
        mid=self.height()/2
        for i,b in enumerate(bins):
            painter.drawLine(int(i*width/len(bins)),int(mid-float(np.max(b))*mid*.9),
                             int(i*width/len(bins)),int(mid-float(np.min(b))*mid*.9))

class Timeline(QWidget):
    region_clicked=Signal(int)
    def __init__(self):
        super().__init__(); self.turns=[]; self.duration=1; self.setMinimumHeight(180)
    def set_turns(self,turns,duration): self.turns=turns; self.duration=max(1,duration); self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor('#131b29')); colors=['#59c5d9','#ad89ff','#ffbe73','#67d3a0']
        speakers=sorted({t['speaker'] for t in self.turns}); self.hit=[]
        for index,speaker in enumerate(speakers):
            y=index*40+25; p.setPen(QColor('white')); p.drawText(5,y+17,speaker)
            for i,t in enumerate(self.turns):
                if t['speaker']!=speaker: continue
                x=120+int(t['start']/self.duration*(self.width()-130)); w=max(2,int((t['end']-t['start'])/self.duration*(self.width()-130)))
                from PySide6.QtCore import QRect
                rect=QRect(x,y,w,25); p.fillRect(rect,QColor(colors[index%4])); self.hit.append((rect,i))
                if any(o['speaker']!=speaker and o['start']<t['end'] and o['end']>t['start'] for o in self.turns):
                    p.setPen(QPen(QColor('#ff655d'),2)); p.drawRect(rect)
        for i in range(6):
            p.setPen(QColor('#9cacc5')); p.drawText(120+int(i/5*(self.width()-130)),18,f'{i*self.duration/5:.1f}s')
    def mousePressEvent(self,event):
        for rect,index in getattr(self,'hit',[]):
            if rect.contains(event.position().toPoint()): self.region_clicked.emit(index); break
