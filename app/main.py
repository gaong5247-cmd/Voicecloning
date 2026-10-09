import sys, argparse, logging, json
from app.config import DATA, configure_offline, VERSION

def main():
    parser=argparse.ArgumentParser(description='CloneVoice Studio')
    parser.add_argument('--diagnose',action='store_true'); parser.add_argument('--smoke-test',action='store_true')
    parser.add_argument('--download',choices=['voice','realtime','separation','diarization'])
    parser.add_argument('--engine-import-test',action='store_true')
    parser.add_argument('--source'); parser.add_argument('--profile'); parser.add_argument('--output')
    parser.add_argument('--reference'); parser.add_argument('--consent',action='store_true')
    parser.add_argument('--backend',default='auto',choices=['auto','cpu','xpu']); parser.add_argument('--precision',default='fp32',choices=['fp32','fp16','bf16'])
    parser.add_argument('--model',default='quality',choices=['quality','tiny'])
    parser.add_argument('--steps',type=int,default=20); args=parser.parse_args()
    DATA.mkdir(parents=True,exist_ok=True); (DATA/'logs').mkdir(exist_ok=True)
    if sys.stdout is None: sys.stdout=open(DATA/'logs'/'stdout.log','a',encoding='utf-8',buffering=1)
    if sys.stderr is None: sys.stderr=open(DATA/'logs'/'stderr.log','a',encoding='utf-8',buffering=1)
    logging.basicConfig(filename=DATA/'logs'/'app.log',level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    configure_offline()
    logging.info('CloneVoice Studio starting, smoke_test=%s, engine_import_test=%s',args.smoke_test,args.engine_import_test)
    try:
        if args.download:
            import os
            from app.services.models import download_group
            download_group(args.download,os.environ.get('HF_TOKEN'),print); return 0
        if args.engine_import_test:
            from app.config import ROOT
            sys.path.insert(0,str(ROOT/'vendor/seed_vc'))
            import seed_loader
            from modules.flow_matching import CFM
            from modules.length_regulator import InterpolateRegulator
            from modules.bigvgan.bigvgan import BigVGAN
            from modules.campplus.DTDNN import CAMPPlus
            import transformers, speechbrain, pyannote.audio
            from app.config import atomic_json
            atomic_json(DATA/'engine-imports.json',{'status':'passed','version':VERSION,'model_inference':'not executed'})
            return 0
        if args.diagnose:
            from app.optimization.devices import diagnose
            print(json.dumps(diagnose(),indent=2,ensure_ascii=False)); return 0
        if args.reference:
            from app.services.profiles import Profiles
            profile=Profiles().create(args.profile or 'CLI Voice',args.reference,args.consent)
            print(json.dumps(profile,ensure_ascii=False)); return 0
        if args.source:
            if not args.profile or not args.output: parser.error('--source requires --profile ID and --output')
            from app.services.profiles import Profiles
            from app.services.jobs import Job
            from app.engine.seed import SeedEngine
            from app.engine.inference import convert_file
            job=Job(args.source,args.profile,{'steps':args.steps,'chunk_seconds':12,'backend':args.backend,'precision':args.precision,'model_kind':args.model})
            result=convert_file(SeedEngine(args.backend,args.precision,model_kind=args.model),args.source,Profiles().get(args.profile),args.output,job,
                                lambda f,m:print(f'{f:.0%} {m}'))
            print(json.dumps(result['metrics'],indent=2)); return 0
        logging.info('Preparing Qt GUI')
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import QTimer
        from app.gui.main_window import Window
        application=QApplication(sys.argv); application.setApplicationName('CloneVoiceStudio')
        logging.info('Creating main window')
        window=Window(startup_diagnostics=not args.smoke_test); window.show()
        logging.info('Main window displayed')
        if args.smoke_test:
            window.timer.stop()
            def finish():
                if window.worker and window.worker.isRunning(): QTimer.singleShot(100,finish)
                else: window.close(); application.quit()
            QTimer.singleShot(1000,finish)
            logging.info('GUI smoke test exit timer scheduled')
        result=application.exec()
        logging.info('Qt event loop exited with %s',result)
        return result
    except Exception:
        logging.exception('Application error'); raise

if __name__=='__main__': sys.exit(main())
