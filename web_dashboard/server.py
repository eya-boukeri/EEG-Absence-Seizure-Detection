import os
import sys
import json
import re
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import engine

PORT = 5050
STATIC_DIR = os.path.dirname(__file__)

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, X-Filename')
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == '/api/info':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            info = engine.get_channel_info()
            self.wfile.write(json.dumps(info).encode('utf-8'))
            return

        elif path == '/api/datasets':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            datasets = engine.list_available_datasets()
            self.wfile.write(json.dumps(datasets).encode('utf-8'))
            return

        elif path == '/api/pipeline/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            status = engine.get_pipeline_status()
            self.wfile.write(json.dumps(status).encode('utf-8'))
            return

        elif path in ['/api/pipeline/clean', '/api/pipeline/cleaning_summary']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            res = engine.get_cleaning_summary()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        elif path in ['/api/pipeline/extract', '/api/pipeline/features_summary']:
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            res = engine.get_features_summary()
            self.wfile.write(json.dumps(res).encode('utf-8'))
            return

        elif path == '/api/signal':
            channel = query.get('channel', ['EEGFp1'])[0]
            start_sec = float(query.get('start', [0])[0])
            end_sec = query.get('end', [None])[0]
            if end_sec is not None:
                end_sec = float(end_sec)
            max_points = int(query.get('max_points', [2500])[0])

            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            sig_data = engine.get_signal_data(channel, start_sec, end_sec, max_points)
            self.wfile.write(json.dumps(sig_data).encode('utf-8'))
            return

        # Default static file serving
        return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == '/api/upload':
            filename = self.headers.get('X-Filename')
            content_type = self.headers.get('Content-Type', '')
            content_length = int(self.headers.get('Content-Length', 0))

            save_path = None
            if filename:
                filename = os.path.basename(filename)
                save_path = os.path.join(engine.DATA_DIR, filename)
                with open(save_path, 'wb') as f:
                    f.write(self.rfile.read(content_length))
            else:
                body = self.rfile.read(content_length)
                if 'boundary=' in content_type:
                    boundary = content_type.split('boundary=')[1].strip().encode('utf-8')
                    parts = body.split(b'--' + boundary)
                    for part in parts:
                        if b'filename="' in part:
                            fn_match = re.search(rb'filename="([^"]+)"', part)
                            if fn_match:
                                filename = fn_match.group(1).decode('utf-8', errors='ignore')
                                filename = os.path.basename(filename)
                                header_end = part.find(b'\r\n\r\n')
                                if header_end != -1:
                                    file_content = part[header_end + 4:]
                                    if file_content.endswith(b'\r\n'):
                                        file_content = file_content[:-2]
                                    save_path = os.path.join(engine.DATA_DIR, filename)
                                    with open(save_path, 'wb') as f:
                                        f.write(file_content)
                                    break
                if not save_path:
                    filename = "uploaded_eeg.csv"
                    save_path = os.path.join(engine.DATA_DIR, filename)
                    with open(save_path, 'wb') as f:
                        f.write(body)

            file_size_mb = round(os.path.getsize(save_path) / (1024 * 1024), 2)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'success',
                'filename': filename,
                'size_mb': file_size_mb,
                'message': f"Fichier {filename} ({file_size_mb} Mo) enregistré avec succès dans data/"
            }).encode('utf-8'))
            return

        elif parsed.path in ['/api/pipeline/clean']:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length > 0 else "{}"
            try:
                data = json.loads(body)
            except Exception:
                data = {}
            patient_id = data.get('patient_id')
            signal_file = data.get('signal_file')
            annotation_file = data.get('annotation_file')

            try:
                engine.load_data(patient_id=patient_id, signal_file=signal_file, annotation_file=annotation_file)
                res = engine.get_cleaning_summary()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        elif parsed.path in ['/api/pipeline/extract']:
            try:
                res = engine.get_features_summary()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        elif parsed.path in ['/api/pipeline/reset']:
            engine.reset_cache()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'reset_done'}).encode('utf-8'))
            return

        elif parsed.path in ['/api/train', '/api/pipeline/train']:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            try:
                data = json.loads(body)
            except Exception:
                data = {}

            channel = data.get('channel', 'ALL')
            model_name = data.get('model', 'Random Forest')
            use_smote = data.get('use_smote', True)
            use_temporal_filter = data.get('use_temporal_filter', True)

            try:
                res = engine.train_and_evaluate(channel, model_name, use_smote=use_smote, use_temporal_filter=use_temporal_filter)
                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(res).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            return

        self.send_response(404)
        self.end_headers()

def run(port=PORT):
    print("Chargement initial des données EEG...")
    engine.load_data()
    print("Préchauffage du modèle multi-canaux (294 descripteurs avec ratios spectraux)...")
    engine.extract_all_features()
    engine.train_and_evaluate('ALL', 'Random Forest')
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    print("\n========================================================")
    print(f"Serveur Dashboard EEG démarré sur : http://localhost:{port}")
    print("========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nArrêt du serveur.")
        httpd.server_close()

if __name__ == '__main__':
    run()
