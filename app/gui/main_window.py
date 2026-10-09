import json, logging, traceback, time
from pathlib import Path
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication,QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,
    QPushButton,QLabel,QListWidget,QStackedWidget,QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QCheckBox,
    QTextEdit,QProgressBar,QFileDialog,QMessageBox,QTableWidget,QTableWidgetItem,QSplitter,QScrollArea)
from app.config import DATA, VERSION, atomic_json
from app.services.profiles import Profiles
from app.gui.widgets import Waveform, Timeline

class Worker(QThread):
    progress=Signal(float,str); done=Signal(object); failed=Signal(str)
    def __init__(self,task): super().__init__(); self.task=task
    def run(self):
        try: self.done.emit(self.task(lambda f,m:self.progress.emit(float(f),str(m))))
        except Exception as exc:
            logging.exception('Worker failed'); self.failed.emit(type(exc).__name__+': '+str(exc))

def button(text,action):
    b=QPushButton(text); b.clicked.connect(action); return b

def spin(minimum,maximum,value):
    b=QSpinBox(); b.setRange(minimum,maximum); b.setValue(value); return b

class Window(QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('CloneVoice Studio • '+VERSION); self.resize(1280,850)
        self.profiles=Profiles(); self.engine=None; self.engine_key=None; self.worker=None
        self.active_job=None; self.rt=None; self.project=None; self.last_output=None
        config_path=DATA/'settings.json'
        try: self.settings=json.loads(config_path.read_text()) if config_path.exists() else {'theme':'dark','language':'한국어'}
        except (ValueError,OSError):
            logging.exception('Invalid settings; restoring defaults'); self.settings={'theme':'dark','language':'한국어'}
        central=QWidget(); self.setCentralWidget(central); outer=QVBoxLayout(central)
        top=QHBoxLayout(); top.addWidget(QLabel('CLONEVOICE STUDIO   /   Local Voice Workspace'))
        top.addStretch(); self.language=QComboBox(); self.language.addItems(['한국어','English']); self.language.setCurrentText(self.settings.get('language','한국어'))
        top.addWidget(self.language); outer.addLayout(top)
        body=QSplitter(); self.nav=QListWidget(); self.nav.setMaximumWidth(245); self.pages=QStackedWidget()
        self.names=['Dashboard','Realtime Voice Conversion','Advanced Inference','Multi-Speaker Studio','Voice Profiles',
                    'Audio Devices','Model Manager','XPU Performance','Settings','Diagnostics']
        self.korean=['대시보드','실시간 음성 변환','파일 음성 변환','다중 화자 스튜디오','음성 프로필','오디오 장치','모델 관리','XPU 성능','설정','진단']
        self.nav.addItems(self.names); body.addWidget(self.nav); body.addWidget(self.pages); outer.addWidget(body,1)
        self.log=QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(130); outer.addWidget(self.log)
        self.progress=QProgressBar(); outer.addWidget(self.progress)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.build_dashboard(); self.build_realtime(); self.build_inference(); self.build_studio(); self.build_profiles()
        self.build_devices(); self.build_models(); self.build_performance(); self.build_settings(); self.build_diagnostics()
        self.nav.setCurrentRow(0); self.refresh_profiles(); self.apply_theme()
        self.language.currentTextChanged.connect(self.translate); self.translate()
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(500)
        QTimer.singleShot(100,self.run_diagnostics)
    def page(self,title,description):
        w=QWidget(); layout=QVBoxLayout(w); title_label=QLabel(title); title_label.setStyleSheet('font-size:24px;font-weight:700')
        layout.addWidget(title_label); label=QLabel(description); label.setWordWrap(True); layout.addWidget(label)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(w); self.pages.addWidget(scroll)
        return layout
    def translate(self):
        ko=self.language.currentText()=='한국어'
        for i in range(10): self.nav.item(i).setText(self.korean[i] if ko else self.names[i])
        self.settings['language']=self.language.currentText(); self.save_settings()
        # Controls retain bilingual labels; technical model names remain unchanged.
    def save_settings(self): atomic_json(DATA/'settings.json',self.settings)
    def apply_theme(self):
        dark=self.settings.get('theme')=='dark'; bg='#0c1320' if dark else '#f4f6fa'; fg='#e3eaf5' if dark else '#172333'
        self.setStyleSheet(f'QWidget {{background:{bg};color:{fg};font-family:"Segoe UI","Malgun Gothic";font-size:13px;}}'
            'QPushButton {background:#265b77;color:white;border:0;border-radius:6px;padding:9px 16px;}'
            'QPushButton:disabled {background:#344153;color:#8593a7;}'
            'QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QTextEdit,QListWidget,QTableWidget {border:1px solid #39455c;border-radius:4px;padding:5px;}'
            'QListWidget::item {padding:11px;} QListWidget::item:selected {background:#265b77;}')
    def message(self,text): self.log.append(text)
    def task(self,func,done=None):
        if self.worker and self.worker.isRunning(): self.message('A task is already running / 작업 중임'); return
        if self.rt and (not self.rt.stop_event.is_set() or (getattr(self.rt,'worker',None) and self.rt.worker.is_alive())):
            self.message('Stop realtime first / 실시간 변환부터 중지'); return
        self.progress.setValue(0); self.worker=Worker(func)
        self.worker.progress.connect(lambda f,m:(self.progress.setValue(int(f*100)),self.message(m)))
        self.worker.failed.connect(self.error); self.worker.done.connect(done or (lambda value:self.message('Complete / 완료')))
        self.worker.start()
    def error(self,text):
        self.message(text); QMessageBox.warning(self,'작업 실패 / Task failed',text+'\n\nDiagnostics 로그 확인 · 모델 다운로드/파일/장치/출력 경로를 확인하세요')
    def engine_options(self):
        return (self.backend.currentText(),self.precision.currentText(),self.compile.isChecked(),self.model_kind.currentText())
    def ensure_engine(self,key):
        from app.engine.seed import SeedEngine
        if key!=self.engine_key:
            if self.engine: self.engine.unload()
            self.engine=SeedEngine(*key); self.engine_key=key
        return self.engine
    def choose(self,edit,save=False,filter='Audio / Video (*.wav *.mp3 *.flac *.ogg *.m4a *.mp4 *.mkv)'):
        path=(QFileDialog.getSaveFileName if save else QFileDialog.getOpenFileName)(self,'Select file',edit.text(),filter)[0]
        if path: edit.setText(path)
    def file_row(self,layout,label,save=False):
        row=QHBoxLayout(); edit=QLineEdit(); edit.setPlaceholderText(label); row.addWidget(edit)
        row.addWidget(button('Browse / 찾아보기',lambda:self.choose(edit,save))); layout.addLayout(row); return edit
    def build_dashboard(self):
        l=self.page('CloneVoice Studio','Zero-shot • Seed-VC V1 • Local processing • Intel XPU / CPU')
        l.addWidget(QLabel('1  Model Manager에서 음성 변환 모델 다운로드\n2  Voice Profiles에서 권한 확인 후 참조 음성 등록\n3  Advanced Inference에서 원본 파일을 WAV로 변환\n4  실시간: CABLE Input으로 출력 → 채팅 앱 마이크를 CABLE Output으로 설정'))
        l.addWidget(QLabel('PREVIEW: Intel hardware, Windows audio and multilingual quality require device validation.\nOpenVINO / NPU / INT8 / 3-speaker separation are unavailable.\nV1 realtime uses buffered inference; sub-100ms latency is not claimed.'))
        self.dashboard_status=QLabel('Diagnostics pending'); self.dashboard_status.setWordWrap(True); l.addWidget(self.dashboard_status); l.addStretch()
    def build_inference(self):
        l=self.page('Advanced Inference / 파일 변환','원본을 수정하지 않고 모델 결과를 WAV로 저장함. 영상은 오디오 추출 후 별도 Remux 가능.')
        self.source=self.file_row(l,'Source / 원본'); self.waveform=Waveform(); l.addWidget(self.waveform)
        l.addWidget(button('Inspect / 파형 보기',self.inspect))
        form=QFormLayout(); self.profile_combo=QComboBox(); form.addRow('Voice Profile / 목표 음성',self.profile_combo)
        self.mode=QComboBox(); self.mode.addItems(['FAST','BALANCED','QUALITY']); self.mode.setCurrentText('BALANCED'); form.addRow('Preset',self.mode)
        self.steps=spin(1,100,20); form.addRow('Diffusion Steps',self.steps)
        self.chunk=spin(1,18,12); form.addRow('Chunk seconds',self.chunk)
        self.backend=QComboBox(); self.backend.addItems(['auto','xpu','cpu']); form.addRow('Backend',self.backend)
        self.model_kind=QComboBox(); self.model_kind.addItems(['quality','tiny']); form.addRow('Seed model',self.model_kind)
        self.precision=QComboBox(); self.precision.addItems(['fp32','fp16','bf16']); form.addRow('Precision',self.precision)
        self.compile=QCheckBox('Experimental torch.compile (compiler needed; unverified)'); form.addRow(self.compile)
        unsupported=QLabel('Pitch / F0 / Formant / Batch: disabled — current 22k V1 model is not F0-conditioned.\nNoise reduction and loudness normalization are not implemented. Output peak limiting is enabled.')
        unsupported.setWordWrap(True); form.addRow(unsupported); l.addLayout(form)
        self.mode.currentTextChanged.connect(self.preset)
        self.output=self.file_row(l,'Output WAV / 출력',True); self.output.setText(str(DATA/'output'/'converted.wav'))
        row=QHBoxLayout(); row.addWidget(button('Convert / 변환',self.convert)); row.addWidget(button('Cancel / 취소',self.cancel))
        row.addWidget(button('Resume Session / 이어하기',self.resume)); row.addWidget(button('Listen / 미리 듣기',self.listen))
        row.addWidget(button('Remux video / 영상 합치기',self.remux)); l.addLayout(row); l.addStretch()
    def preset(self,name):
        from app.engine.inference import PRESETS
        self.steps.setValue(PRESETS[name]['steps']); self.chunk.setValue(PRESETS[name]['chunk_seconds'])
    def inspect(self):
        source=self.source.text()
        def work(p):
            from app.audio_io.files import read_audio
            wave,sr=read_audio(source); return (wave,sr)
        self.task(work,lambda result:(self.waveform.set_wave(result[0]),self.message(f'{len(result[0])/result[1]:.2f}s, {result[1]}Hz mono')))
    def convert(self):
        from app.services.jobs import Job
        source=self.source.text(); ident=self.profile_combo.currentData(); output=self.output.text()
        if not ident or not source: self.error('Select source and Voice Profile'); return
        settings={'steps':self.steps.value(),'chunk_seconds':self.chunk.value(),'backend':self.backend.currentText(),'precision':self.precision.currentText()}
        job=Job(source,ident,settings); self.active_job=job
        self.run_conversion(job,output)
    def run_conversion(self,job,output):
        key=self.engine_options()
        def work(progress):
            from app.engine.inference import convert_file
            return convert_file(self.ensure_engine(key),job.record['source'],self.profiles.get(job.record['profile']),output,job,progress)
        self.task(work,lambda r:(setattr(self,'last_output',output),self.message(json.dumps(r['metrics'],ensure_ascii=False,indent=2))))
    def resume(self):
        path=QFileDialog.getOpenFileName(self,'Resume session',str(DATA/'sessions'),'Session (session.json)')[0]
        if not path: return
        from app.services.jobs import Job
        try:
            job=Job.resume(path); self.active_job=job; self.source.setText(job.record['source'])
            settings=job.record['settings']; self.backend.setCurrentText(settings['backend']); self.precision.setCurrentText(settings['precision'])
            self.run_conversion(job,job.record.get('output',self.output.text()))
        except Exception as exc: self.error(str(exc))
    def cancel(self):
        if self.active_job: self.active_job.cancelled.set(); self.message('Cancel requested after current model block / 현재 블록 종료 후 취소')
        if self.worker: self.worker.requestInterruption()
    def listen(self):
        path=self.last_output or self.output.text()
        if Path(path).exists(): QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    def remux(self):
        video=self.source.text(); audio=self.last_output or self.output.text()
        output=QFileDialog.getSaveFileName(self,'Save video','','MP4 (*.mp4);;MKV (*.mkv)')[0]
        if output:
            def work(p):
                from app.audio_io.files import remux
                remux(video,audio,output)
            self.task(work)
    def build_profiles(self):
        l=self.page('Voice Profiles / 음성 프로필','5–25초의 깨끗한 단일 화자 권장. 에너지 기반 발화 추출 사용; 다른 화자 여부는 자동 검증하지 않음.')
        self.profile_name=QLineEdit(); self.profile_name.setPlaceholderText('Profile name'); l.addWidget(self.profile_name)
        self.reference=self.file_row(l,'Reference audio / 참조 음성')
        self.consent=QCheckBox('본인 소유 또는 사용 허락 받은 음성임 / I own or have permission to use this voice'); l.addWidget(self.consent)
        l.addWidget(button('Create Profile / 등록',self.create_profile)); self.profile_list=QTextEdit(); self.profile_list.setReadOnly(True); l.addWidget(self.profile_list)
    def create_profile(self):
        name=self.profile_name.text(); ref=self.reference.text(); consent=self.consent.isChecked()
        self.task(lambda p:self.profiles.create(name,ref,consent),lambda value:(self.refresh_profiles(),self.message('Profile created: '+value['name'])))
    def refresh_profiles(self):
        entries=self.profiles.list(); current=self.profile_combo.currentData()
        for combo in (self.profile_combo,self.rt_profile):
            combo.clear()
            for value in entries: combo.addItem(value['name'],value['id'])
        if current: self.profile_combo.setCurrentIndex(self.profile_combo.findData(current))
        self.profile_list.setPlainText(json.dumps(entries,indent=2,ensure_ascii=False))
    def build_realtime(self):
        l=self.page('Realtime / 실시간 변환','Tiny XLSR/HiFT preview: realtime 모델 다운로드 필요. 장치 실측 전에는 지연 보장 없음. 가상 오디오 장치는 별도 설치해야 함. 오디오 콜백에서는 추론하지 않음.')
        form=QFormLayout(); self.input_device=QComboBox(); self.output_device=QComboBox(); self.rt_profile=QComboBox()
        form.addRow('Microphone / 마이크',self.input_device); form.addRow('Output / 출력',self.output_device); form.addRow('Profile',self.rt_profile)
        self.rt_sr=QComboBox(); self.rt_sr.addItems(['48000','44100']); form.addRow('Sample rate',self.rt_sr)
        self.rt_chunk=QDoubleSpinBox(); self.rt_chunk.setRange(.5,4); self.rt_chunk.setSingleStep(.1); self.rt_chunk.setValue(1); form.addRow('Block seconds',self.rt_chunk)
        self.rt_steps=spin(2,40,6); form.addRow('Steps',self.rt_steps); self.rt_gate=spin(-80,0,-45); form.addRow('Gate dB',self.rt_gate)
        self.rt_gain=QDoubleSpinBox(); self.rt_gain.setRange(0,4); self.rt_gain.setValue(1); form.addRow('Gain',self.rt_gain)
        self.rt_mute=QCheckBox('Mute / 음소거'); form.addRow(self.rt_mute)
        self.rt_original=QCheckBox('Monitor original / 원본 출력 (이어폰 사용)'); form.addRow(self.rt_original)
        l.addLayout(form); row=QHBoxLayout(); row.addWidget(button('Start / 시작',self.start_rt)); row.addWidget(button('Stop / 중지',self.stop_rt)); l.addLayout(row)
        self.rt_stats=QTextEdit(); self.rt_stats.setReadOnly(True); l.addWidget(self.rt_stats)
    def start_rt(self):
        from app.audio_io.realtime import Realtime
        ident=self.rt_profile.currentData()
        if not ident: self.error('Create a Voice Profile first'); return
        profile=self.profiles.get(ident); inp=self.input_device.currentData(); out=self.output_device.currentData()
        sr=int(self.rt_sr.currentText()); chunk=self.rt_chunk.value(); steps=self.rt_steps.value(); gate=self.rt_gate.value(); gain=self.rt_gain.value()
        key=(*self.engine_options()[:3],'tiny')
        def work(p):
            self.rt=Realtime(self.ensure_engine(key),profile,inp,out,sr,chunk,steps,gate,gain); self.rt.start(); return self.rt
        self.task(work,lambda r:self.message('Realtime stream started; use headphones / 이어폰 권장'))
    def stop_rt(self):
        if self.rt:
            self.rt.stop()
            if getattr(self.rt,'worker',None) and self.rt.worker.is_alive():
                self.message('Realtime stopping after current block; wait before starting another task')
            else: self.message('Realtime stopped'); self.rt=None
    def build_devices(self):
        l=self.page('Audio Devices / 오디오 장치','CABLE Input은 프로그램 출력 장치, CABLE Output은 카카오톡·Discord의 마이크 장치임.')
        l.addWidget(button('Refresh / 새로고침',self.refresh_devices)); self.devices_text=QTextEdit(); self.devices_text.setReadOnly(True); l.addWidget(self.devices_text)
        QTimer.singleShot(50,self.refresh_devices)
    def refresh_devices(self):
        try:
            import sounddevice as sd
            devices=sd.query_devices(); self.input_device.clear(); self.output_device.clear()
            for i,d in enumerate(devices):
                label=f'{i}: {d["name"]} ({d["default_samplerate"]:.0f} Hz)'
                if d['max_input_channels']: self.input_device.addItem(label,i)
                if d['max_output_channels']: self.output_device.addItem(label,i)
            self.devices_text.setPlainText(str(devices))
        except Exception as exc: self.devices_text.setPlainText(str(exc))
    def build_models(self):
        l=self.page('Model Manager / 모델 관리','명시적으로 다운로드한 모델만 로컬 추론에 사용함. 오디오 파일은 업로드하지 않음.')
        self.model_group=QComboBox(); self.model_group.addItems(['voice','realtime','separation','diarization']); l.addWidget(self.model_group)
        self.token=QLineEdit(); self.token.setEchoMode(QLineEdit.Password); self.token.setPlaceholderText('Hugging Face token (not saved) — Community-1 requires model access consent'); l.addWidget(self.token)
        self.model_agree=QCheckBox('모델 라이선스/접근 조건을 확인했음 / Reviewed model licenses and access terms'); l.addWidget(self.model_agree)
        l.addWidget(button('Community-1 terms',lambda:QDesktopServices.openUrl(QUrl('https://huggingface.co/pyannote/speaker-diarization-community-1'))))
        l.addWidget(button('Download / 다운로드',self.download)); l.addWidget(button('Verify SHA256 / 무결성 확인',self.verify_models))
        l.addWidget(QLabel('Seed-VC GPL-3.0 · Whisper Apache-2.0 · BigVGAN MIT\nCAMPPlus / SpeechBrain Apache-2.0 · Community-1 CC-BY-4.0\npyannote / SepFormer run on CPU in this preview; XPU compatibility is unverified.')); l.addStretch()
    def download(self):
        if not self.model_agree.isChecked(): self.error('Review model conditions and check the confirmation'); return
        group=self.model_group.currentText(); token=self.token.text() or None; self.token.clear()
        def work(p):
            from app.services.models import download_group
            def check():
                if QThread.currentThread().isInterruptionRequested(): raise RuntimeError('Download cancelled')
            return download_group(group,token,lambda m:p(0,m),check)
        self.task(work)
    def verify_models(self):
        group=self.model_group.currentText()
        def work(p):
            from app.services.models import check_group
            check_group(group)
        self.task(work)
    def build_performance(self):
        l=self.page('XPU Performance','음성 변환 전체 파이프라인 RTF 기록. GPU 이용률·하드웨어 종단 지연은 측정 경로가 없으면 표시하지 않음.')
        self.performance=QTextEdit(); self.performance.setReadOnly(True); l.addWidget(self.performance)
        l.addWidget(button('Benchmark same input / CPU-XPU 비교',self.benchmark))
    def benchmark(self):
        ident=self.profile_combo.currentData(); source=self.source.text()
        if not ident or not source: self.error('Select reference and source on Advanced Inference'); return
        steps=self.steps.value(); model_kind=self.model_kind.currentText()
        def work(p):
            from app.engine.seed import SeedEngine
            from app.audio_io.files import read_audio
            import torch
            if self.engine: self.engine.unload(); self.engine=None; self.engine_key=None
            wave,sr=read_audio(source); wave=wave[:sr*3]; results=[]; profile=self.profiles.get(ident)
            choices=[('cpu','fp32')]+([('xpu',prec) for prec in ['fp32','fp16','bf16']] if torch.xpu.is_available() else [])
            for backend,prec in choices:
                p(0,backend+' '+prec); engine=SeedEngine(backend,prec,model_kind=model_kind)
                try:
                    first=time.perf_counter(); engine.convert(wave,profile,steps); first=time.perf_counter()-first
                    engine.convert(wave,profile,steps)
                    results.append({'requested_backend':backend,'requested_precision':prec,'first_result_seconds':first,**engine.stats})
                except Exception as exc: results.append({'backend':backend,'precision':prec,'error':str(exc)})
                finally: engine.unload()
            record={'source':source,'seconds':len(wave)/sr,'steps':steps,'results':results,'quality':'not evaluated','openvino':'unavailable'}
            atomic_json(DATA/'benchmark.json',record); return record
        self.task(work,lambda r:self.message(json.dumps(r,indent=2,ensure_ascii=False)))
    def build_settings(self):
        l=self.page('Settings / 설정','모델 다운로드 외 네트워크 기능 없음. 원본을 수정하지 않음.')
        theme=QComboBox(); theme.addItems(['dark','light']); theme.setCurrentText(self.settings['theme'])
        theme.currentTextChanged.connect(lambda value:(self.settings.update(theme=value),self.save_settings(),self.apply_theme()))
        l.addWidget(theme); l.addWidget(QLabel('Data: '+str(DATA))); l.addWidget(button('Open data folder',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(DATA))))); l.addStretch()
    def build_diagnostics(self):
        l=self.page('Diagnostics / 진단','실제 장치 탐지 값과 오류 로그만 표시함.')
        l.addWidget(button('Run diagnostics / 진단 실행',self.run_diagnostics)); self.diagnostics=QTextEdit(); self.diagnostics.setReadOnly(True); l.addWidget(self.diagnostics)
    def run_diagnostics(self):
        def work(p):
            from app.optimization.devices import diagnose
            return diagnose()
        self.task(work,lambda result:(self.diagnostics.setPlainText(json.dumps(result,indent=2,ensure_ascii=False)),self.dashboard_status.setText('XPU: '+str(result['xpu_available'])+'  '+result.get('device','CPU'))))
    def build_studio(self):
        l=self.page('Multi-Speaker Studio','Community-1 → overlap-only SepFormer 2-source → ECAPA matching → track assignment → mix. 3명 중첩은 원본 유지 후 검토 필요.')
        self.studio_source=self.file_row(l,'Studio source'); row=QHBoxLayout()
        row.addWidget(button('New project / 새 프로젝트',self.new_project)); row.addWidget(button('Open project',self.open_project));
        row.addWidget(button('Diarize / 화자 분석',self.diarize)); row.addWidget(button('Separate / 중첩 분리',self.separate)); l.addLayout(row)
        self.timeline=Timeline(); self.timeline.region_clicked.connect(lambda i:self.turn_table.selectRow(i)); l.addWidget(self.timeline)
        self.turn_table=QTableWidget(0,3); self.turn_table.setHorizontalHeaderLabels(['Start (s)','End (s)','Speaker ID']); l.addWidget(self.turn_table)
        row=QHBoxLayout(); row.addWidget(button('Add turn / 구간 추가',self.add_turn)); row.addWidget(button('Delete turn',self.delete_turn)); row.addWidget(button('Save edits / 구간 저장',self.save_turns)); l.addLayout(row)
        self.assignment=QTableWidget(0,3); self.assignment.setHorizontalHeaderLabels(['Speaker','Voice Profile','Gain']); l.addWidget(self.assignment)
        self.review_select=QComboBox(); l.addWidget(self.review_select)
        row=QHBoxLayout(); row.addWidget(button('Listen source 0',lambda:self.preview_overlap(0))); row.addWidget(button('Listen source 1',lambda:self.preview_overlap(1)))
        row.addWidget(button('Match in order',lambda:self.resolve_overlap(False,False))); row.addWidget(button('Swap sources',lambda:self.resolve_overlap(True,False)))
        row.addWidget(button('Keep original overlap',lambda:self.resolve_overlap(False,True))); l.addLayout(row)
        l.addWidget(button('Render assigned voices / 변환 및 믹싱',self.render_studio)); self.studio_status=QTextEdit(); self.studio_status.setReadOnly(True); l.addWidget(self.studio_status)
    def new_project(self):
        directory=QFileDialog.getExistingDirectory(self,'Project folder',str(DATA))
        if directory:
            from app.engine.studio import Studio
            self.project=Studio(directory,self.studio_source.text()); self.refresh_studio()
    def open_project(self):
        file=QFileDialog.getOpenFileName(self,'Open project',str(DATA),'Project (project.json)')[0]
        if file:
            try:
                from app.engine.studio import Studio
                self.project=Studio(Path(file).parent); self.studio_source.setText(self.project.data['source']); self.refresh_studio()
            except Exception as exc: self.error(str(exc))
    def refresh_studio(self):
        if not self.project: return
        self.review_select.clear()
        for i,r in enumerate(self.project.data['review']): self.review_select.addItem(f'{i}: {r["start"]:.2f}–{r["end"]:.2f}s {r["speakers"]} {r.get("resolved","review required")}',i)
        turns=self.project.data['turns']; self.turn_table.setRowCount(len(turns))
        for i,t in enumerate(turns):
            for j,key in enumerate(('start','end','speaker')): self.turn_table.setItem(i,j,QTableWidgetItem(str(t[key])))
        self.timeline.set_turns(turns,max([t['end'] for t in turns] or [1]))
        speakers=sorted({t['speaker'] for t in turns}); self.assignment.setRowCount(len(speakers))
        for i,s in enumerate(speakers):
            self.assignment.setItem(i,0,QTableWidgetItem(s)); combo=QComboBox(); combo.addItem('Keep Original',None)
            for profile in self.profiles.list(): combo.addItem(profile['name'],profile['id'])
            old=self.project.data['assignments'].get(s,{}); combo.setCurrentIndex(max(0,combo.findData(old.get('profile'))))
            self.assignment.setCellWidget(i,1,combo); gain=QDoubleSpinBox(); gain.setRange(0,4); gain.setValue(old.get('gain',1)); self.assignment.setCellWidget(i,2,gain)
        self.studio_status.setPlainText(json.dumps({'review':self.project.data['review'],'tracks':self.project.data['tracks']},indent=2,ensure_ascii=False))
    def add_turn(self):
        i=self.turn_table.rowCount(); self.turn_table.insertRow(i)
        for j,value in enumerate(['0','1','SPEAKER_00']): self.turn_table.setItem(i,j,QTableWidgetItem(value))
    def delete_turn(self):
        row=self.turn_table.currentRow()
        if row>=0: self.turn_table.removeRow(row)
    def save_turns(self):
        if not self.project: self.error('Create or open a project'); return False
        try:
            turns=[]
            for i in range(self.turn_table.rowCount()):
                a=float(self.turn_table.item(i,0).text()); b=float(self.turn_table.item(i,1).text()); s=self.turn_table.item(i,2).text().strip()
                if not 0<=a<b or not s or not s.replace('_','').isalnum(): raise ValueError('Invalid turn or Speaker ID')
                turns.append({'start':a,'end':b,'speaker':s})
            if turns!=self.project.data['turns']: self.project.data['tracks']={}; self.project.data['review']=[]
            self.project.data['turns']=turns; self.project.save(); self.refresh_studio(); return True
        except Exception as exc: self.error(str(exc)); return False
    def diarize(self):
        if not self.project: self.error('Create a project first'); return
        project=self.project
        def work(p):
            if self.engine: self.engine.unload()
            return project.diarize(lambda m:p(0,m))
        self.task(work,lambda r:self.refresh_studio())
    def separate(self):
        if not self.save_turns(): return
        project=self.project
        def work(p):
            if self.engine: self.engine.unload()
            return project.split_tracks(lambda m:p(0,m),lambda:self.check_cancel())
        self.task(work,lambda r:self.refresh_studio())
    def check_cancel(self):
        if QThread.currentThread().isInterruptionRequested(): raise RuntimeError('Cancelled')
    def preview_overlap(self,source):
        if not self.project or self.review_select.currentData() is None: return
        try: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.preview_source(self.review_select.currentData(),source))))
        except Exception as exc: self.error(str(exc))
    def resolve_overlap(self,swap,keep):
        if not self.project or self.review_select.currentData() is None: return
        project=self.project; index=self.review_select.currentData()
        self.task(lambda p:project.resolve_overlap(index,swap,keep),lambda r:self.refresh_studio())
    def render_studio(self):
        if not self.project or not self.project.data['tracks']: self.error('Generate tracks first'); return
        project=self.project
        for i in range(self.assignment.rowCount()):
            speaker=self.assignment.item(i,0).text(); project.data['assignments'][speaker]={
                'profile':self.assignment.cellWidget(i,1).currentData(),'gain':self.assignment.cellWidget(i,2).value()}
        project.save(); output=QFileDialog.getSaveFileName(self,'Output WAV','','WAV (*.wav)')[0]
        if output:
            key=self.engine_options(); steps=self.steps.value()
            def work(p): return project.render(self.ensure_engine(key),self.profiles,output,steps,lambda m:p(0,m),self.check_cancel)
            self.task(work,lambda r:(setattr(self,'last_output',output),self.refresh_studio(),self.message('Studio output saved')))
    def tick(self):
        if self.rt:
            self.rt.muted=self.rt_mute.isChecked(); self.rt.original=self.rt_original.isChecked()
            self.rt_stats.setPlainText(json.dumps(self.rt.metrics,indent=2,ensure_ascii=False))
        info={'engine':self.engine.stats if self.engine else None,'gpu_utilization':None,'hardware_latency':None}
        try:
            import psutil
            info.update(cpu_percent=psutil.cpu_percent(),system_memory_available=psutil.virtual_memory().available)
        except ImportError: pass
        self.performance.setPlainText(json.dumps(info,indent=2,ensure_ascii=False))
    def closeEvent(self,event):
        if self.worker and self.worker.isRunning():
            self.cancel(); self.message('Waiting for worker to finish safely; close again after task stops.'); event.ignore(); return
        self.stop_rt(); self.save_settings(); event.accept()
