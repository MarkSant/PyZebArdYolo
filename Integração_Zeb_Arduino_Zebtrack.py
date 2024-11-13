import cv2
import numpy as np
import serial
from ultralytics import YOLO
import os
import csv
from tkinter import Tk, filedialog, simpledialog, messagebox, Button, Label, Frame, StringVar, OptionMenu, Toplevel
import time
import threading
import queue
import logging
from datetime import datetime

class FrameManager:
    def __init__(self):
        self.frame_count = 0
        self.processed_frames = set()
        self.recording_start_time = None
        self.frame_copy_for_recording = None  # Atributo para armazenar a cópia do frame para gravação
        self.frame_copy_for_detection = None  # Atributo para armazenar a cópia do frame para detecção
        
    def register_frame(self, frame_count, is_processed=False):
        if is_processed:
            self.processed_frames.add(frame_count)
            
    def start_recording(self):
        self.recording_start_time = time.time()
        self.frame_count = 0
        self.processed_frames.clear()
        
    def get_timestamp(self):
        if self.recording_start_time is None:
            return 0
        return time.time() - self.recording_start_time
    
    def get_frame_info(self, frame_count):
        return {
            'was_processed': frame_count in self.processed_frames,
            'timestamp': self.get_timestamp()
        }
    
    def store_frame_copy_for_recording(self, frame):
        frame_copy = frame.copy()  # Faça a cópia fora do lock
        with lock:
            self.frame_copy_for_recording = frame_copy


    def store_frame_copy_for_detection(self, frame):
        frame_copy = frame.copy()  # Faça a cópia fora do lock
        with lock:
            self.frame_copy_for_detection = frame_copy

    def get_frame_copy_for_recording(self):
        """
        Retorna a cópia do frame armazenada para gravação.
        """
        return self.frame_copy_for_recording

    def get_frame_copy_for_detection(self):
        """
        Retorna a cópia do frame armazenada para detecção.
        """
        return self.frame_copy_for_detection
    

# Inicializar o frame_manager logo após a classe
frame_manager = FrameManager()

lock = threading.Lock()

# Configuração do sistema de logging
def setup_logging(project_folder):
    log_filename = os.path.join(project_folder, f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def log_frame_processing(frame_count, timestamp, detection_info, queue_sizes):
    logging.info(f"Frame {frame_count}: Timestamp={timestamp:.3f}, "
                f"Detected={bool(detection_info)}, "
                f"Queue_sizes: video={queue_sizes['video']}, "
                f"detection={queue_sizes['detection']}")

# Variáveis globais
video_queue = queue.Queue()
detection_queue = queue.Queue()
capturing = False  # Inicialmente não estamos capturando
processing = True
detection_complete_called = False

# Configuração da câmera USB
camera_index = 1  # Índice da câmera USB
cap = cv2.VideoCapture(camera_index)

# Definir a resolução desejada para a Logitech C270 HD Webcam
desired_width = 1280  # Largura máxima suportada pela câmera
desired_height = 720  # Altura máxima suportada pela câmera

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Capturar um frame para obter as dimensões reais
ret, frame = cap.read()
if not ret:
    print("Falha ao capturar imagem da câmera")
    cap.release()
    exit()

actual_height, actual_width = frame.shape[:2]
print(f"Resolução real do frame: {actual_width} x {actual_height}")

arduino_port = 'COM9'  # Modify this with the appropriate serial port
baud_rate = 9600  # Modify this with the appropriate baud rate

confThreshold = 0.3
nmsThreshold = 0.3

crossed_in = False
crossed_out = False
flag = 0
current_square = 0

# Coordinates for lines forming squares
squares = [
    ((150, 490), (360, 660)),  # Quadrado inferior esquerdo
    ((385, 140), (550, 310)),  # Quadrado superior esquerdo
    ((630, 490), (765, 660)),  # Quadrado inferior direito
    ((850, 140), (1020, 310))  # Quadrado superior direito
]

# Colors for the squares
colors = [
    (0, 0, 255),    # Vermelho
    (255, 0, 0),    # Azul
    (0, 255, 0),    # Verde
    (0, 0, 255)     # Vermelho
]

# Commands for entering and exiting each square
enter_commands = [1, 3, 5, 7]
exit_commands = [2, 4, 6, 8]

# Polygon coordinates
polygon = np.array([
    [150, 310],
    [385, 310],
    [385, 140],
    [550, 140],
    [550, 310],
    [850, 310],
    [850, 140],
    [1020, 140],
    [1020, 490],
    [765, 490],
    [765, 660],
    [630, 660],
    [630, 490],
    [360, 490],
    [360, 660],
    [150, 660]
], np.int32)

# Initialize Tkinter
root = Tk()
root.title("Controle de Gravação")

processing_status = StringVar()
processing_status_label = Label(root, textvariable=processing_status)
processing_status_label.pack()

recording = False
out = None
csv_file = None
csv_writer = None
frame_count = 0
fps = 30  # Taxa de quadros alvo
start_time = 0  # Inicialização da variável
stop_event = threading.Event()
recording_start_time = None

# Estabelecer conexão serial com o Arduino Nano
arduino = serial.Serial(arduino_port, baud_rate, timeout=1)

def send_servo_command(boxNumber):
    command = f"{boxNumber}\n"  # Adicionar caractere de nova linha ao comando
    print(f"Enviando comando para Arduino: {command}")  # Para depuração
    arduino.write(command.encode('utf-8'))

def is_inside_square(x1, y1, x2, y2, square):
    (sx1, sy1), (sx2, sy2) = square
    return not (x2 < sx1 or x1 > sx2 or y2 < sy1 or y1 > sy2)

def is_inside_polygon(x1, y1, x2, y2, polygon):
    rect = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], np.int32)
    return cv2.pointPolygonTest(polygon, (x1, y1), False) >= 0 or cv2.pointPolygonTest(polygon, (x2, y2), False) >= 0

# Função findObject
def findObject(outputs, img, csv_writer=None, frame_count=0, timestamp=0):
    global crossed_in, crossed_out, flag, current_square
    
    hT, wT, cT = img.shape
    bbox = []
    confs = []

    for det in outputs:
        x1, y1, x2, y2, confidence = det[:5]
        
        if confidence > confThreshold:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            bbox.append([x1, y1, x2, y2])
            confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    if len(indices) > 0:
        for i in indices:
            i = i[0] if isinstance(i, list) else i  # Lidar com diferentes tipos de retorno de NMSBoxes
            box = bbox[i]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            if is_inside_polygon(x1, y1, x2, y2, polygon):  # Proceder apenas se a bounding box estiver dentro do polígono
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(img, f'{int(confs[i] * 100)}%', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                
                if csv_writer:
                    csv_writer.writerow([timestamp, frame_count, x1, y1, x2, y2, int(confs[i] * 100)])
 
                if flag == 0:
                    for index, square in enumerate(squares):
                        if is_inside_square(x1, y1, x2, y2, square):
                            crossed_in = True
                            flag = 1
                            current_square = index + 1
                            print(f"BoxNumber {enter_commands[index]} ON")
                            send_servo_command(enter_commands[index])
                            break  # Sair do loop uma vez dentro de um quadrado
                elif flag == 1:
                    if not any(is_inside_square(x1, y1, x2, y2, square) for square in squares):
                        crossed_out = True
                        flag = 0
                        print(f"BoxNumber {exit_commands[current_square - 1]} OFF")
                        send_servo_command(exit_commands[current_square - 1])
                        current_square = 0

# Variável global para o intervalo de processamento
processing_interval = 8  # Processar a cada x frames
processing_offset = 2     # Começar no frame y


def frame_capture_thread():
    global cap, stop_event, frame_manager
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Falha ao capturar imagem da câmera no thread de captura")
            break
            
        frame_manager.frame_count += 1
        current_frame_count = frame_manager.frame_count
        
        if processing:
            try:
                detection_frame = frame.copy()
                detection_queue.put((current_frame_count, detection_frame))
                logging.debug(f"Frame {current_frame_count} adicionado à fila de detecção")
            except Exception as e:
                logging.error(f"Erro ao colocar frame na fila de detecção: {e}")
                continue

        if capturing:
            try:
                video_frame = frame.copy()
                video_queue.put((current_frame_count, video_frame))
                logging.debug(f"Frame {current_frame_count} adicionado à fila de vídeo")
            except Exception as e:
                logging.error(f"Erro ao colocar frame na fila de vídeo: {e}")
                continue


def video_recording_thread():
    global out, stop_event, recording
    while not stop_event.is_set() and recording:
        try:
            frame_count, frame = video_queue.get(timeout=1)
            if frame is None:
                logging.warning("Frame é None, não será gravado.")
                continue

            frame_manager.store_frame_copy_for_recording(frame)
            frame_copy = frame_manager.get_frame_copy_for_recording()
            if frame_copy is not None:
                out.write(frame_copy)
            else:
                logging.error("Frame não encontrado para gravação.")
            logging.debug(f"Frame {frame_count} gravado no vídeo")
            
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Erro na gravação do frame: {e}")
            
    if out:
        out.release()
    if csv_file is not None:
        csv_file.close()


# Thread para processamento de objetos e exibição em tempo real
def object_detection_thread():
    global stop_event, frame_manager, detection_complete_called
    
    model = YOLO('C:\\Users\\santa\\OneDrive\\Desktop\\Codigos_Prontos\\best8.pt')

    while not stop_event.is_set():
        try:
            frame_count, frame = detection_queue.get(timeout=1)
            timestamp = frame_manager.get_timestamp()

            queue_sizes = {
                'video': video_queue.qsize(),
                'detection': detection_queue.qsize()
            }

            if (frame_count - processing_offset) % processing_interval == 0:
                frame_manager.store_frame_copy_for_detection(frame)
                frame_copy = frame_manager.get_frame_copy_for_detection()
                if frame_copy is not None:
                    
                    results = model(frame_copy)
                    predictions = results[0].boxes.data.cpu().numpy()
                
                    # Registrar frame processado
                    frame_manager.register_frame(frame_count, True)
                
                    # Logging antes do processamento
                    log_frame_processing(frame_count, timestamp, predictions, queue_sizes)
                
                    # Processar detecções
                    findObject(predictions, frame_copy, csv_writer if recording else None, 
                             frame_count, timestamp)
                else:
                     logging.error("Frame não encontrado para detecção.")
                
                logging.info(f"Processado frame {frame_count} com {len(predictions)} detecções")

            # Desenhar quadrados e polígono
            for i, ((x1, y1), (x2, y2)) in enumerate(squares):
                cv2.rectangle(frame, (x1, y1), (x2, y2), colors[i], 2)
            cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 0), thickness=1)

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
                
        except queue.Empty:
            if not processing and detection_queue.empty():
                logging.info("Detecção concluída.")
                if not detection_complete_called:
                    detection_complete_called = True
                    detection_complete()
                break
            continue
        except Exception as e:
            logging.error(f"Erro no processamento do frame: {e}")


def detection_complete():
    processing_status.set("")  # Limpar o status de processamento
    root.after(0, lambda: messagebox.showinfo("Informação", "Processamento de detecção concluído."))


def wait_for_detection_to_finish():
    global detection_thread
    detection_thread.join()
    detection_complete()

def start_detection_thread():
    global detection_thread, stop_event, detection_complete_called
    stop_event.clear()  # Limpa o stop_event para que a thread de detecção funcione
    detection_complete_called = False
    detection_thread = threading.Thread(target=object_detection_thread)
    detection_thread.start()



# Variável global para armazenar os nomes dos grupos
group_names = []

# Function to create a project
def create_project_button():
    global group_names, project_folder_path
    
    # Abrir uma janela para selecionar uma pasta
    selected_folder = filedialog.askdirectory(title="Selecione a pasta onde criar o novo projeto")
    
    # Se o usuário não selecionou nada, retornar
    if not selected_folder:
        return

    # Pedir ao usuário um nome para a nova pasta
    new_folder_name = simpledialog.askstring("Nome da Pasta", "Digite o nome para a nova pasta:")

    # Se o usuário não fornecer um nome, retornar
    if not new_folder_name:
        return

    # Criar a nova pasta no diretório selecionado
    new_folder_path = os.path.join(selected_folder, new_folder_name)
    
    try:
        os.makedirs(new_folder_path, exist_ok=False)
        messagebox.showinfo("Sucesso", f"Pasta '{new_folder_name}' criada com sucesso em:\n{new_folder_path}")
        project_folder_path = new_folder_path
    except FileExistsError:
        messagebox.showerror("Erro", "A pasta já existe. Tente outro nome.")
    except Exception as e:
        messagebox.showerror("Erro", f"Erro ao criar a pasta: {str(e)}")
    
    # Perguntar ao usuário a quantidade de grupos
    group_count = simpledialog.askinteger("Quantidade de Grupos", "Digite a quantidade de grupos:")
    if group_count is not None:
        group_names = []
        for i in range(group_count):
            group_name = simpledialog.askstring("Nome do Grupo", f"Digite o nome do grupo {i + 1}:")
            if group_name:
                group_names.append(group_name)
        messagebox.showinfo("Informação", f"Você especificou os seguintes grupos: {', '.join(group_names)}")
    else:
        messagebox.showwarning("Atenção", "Nenhuma quantidade de grupos foi especificada.")

# Function to start recording
def start_recording_button():
    global recording, out, csv_file, csv_writer, frame_count, start_time, stop_event, group_names, project_folder_path, base_name, recording_start_time, capturing
    capturing = True  # Iniciar a captura de frames
    
    recording_start_time = time.time()
    start_detection_thread()

    if not recording:
        if not group_names:
            messagebox.showwarning("Aviso", "Nenhum grupo foi criado. Crie um projeto primeiro.")
            return

        # Iniciar ou reiniciar a thread de detecção
        start_detection_thread()

        # Selecionar um grupo
        group_var = StringVar(root)
        group_var.set(group_names[0])  # Definir valor padrão
        group_selection_window = Toplevel(root)
        group_selection_window.title("Selecionar Grupo")

        Label(group_selection_window, text="Selecione o grupo:").pack()
        group_dropdown = OptionMenu(group_selection_window, group_var, *group_names)
        group_dropdown.pack()

        # Botão para confirmar a seleção do grupo
        def confirm_group():
            global out, csv_file, csv_writer, frame_count, recording, start_time, base_name
            cobaia_number = simpledialog.askstring("Número da Cobaia", "Digite o número da cobaia:")
            if not cobaia_number:
                messagebox.showwarning("Aviso", "Você deve fornecer um número para a cobaia.")
                return
            
            # Fechar a janela de seleção de grupo
            group_selection_window.destroy()

            # Definir nome da nova pasta com base no grupo e no número da cobaia
            new_folder_name = f"{group_var.get()}__{cobaia_number}"

            # Criar a nova pasta no diretório do projeto selecionado anteriormente
            new_folder_path = os.path.join(project_folder_path, new_folder_name)

            # Criar a nova pasta
            try:
                os.makedirs(new_folder_path, exist_ok=True)  # Cria a pasta se ainda não existir
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao criar a pasta: {str(e)}")
                return
            
            # Definir o caminho completo do arquivo de vídeo na nova pasta
            out_filename = f"{group_var.get()}_cobaia_{cobaia_number}.mp4"
            out_full_path = os.path.join(new_folder_path, out_filename)
            folder_path = os.path.dirname(out_full_path)

            # Criação do vídeo
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            frame_width, frame_height = actual_width, actual_height
            out = cv2.VideoWriter(out_full_path, fourcc, fps, (frame_width, frame_height))

            if not out.isOpened():
                messagebox.showerror("Erro", "Não foi possível inicializar o gravador de vídeo.")
                return

            # Criação do arquivo CSV com o mesmo nome e localização do vídeo
            # Extrai o nome base do arquivo de vídeo sem a extensão
            base_name = os.path.splitext(os.path.basename(out_full_path))[0]

            # Define o novo nome do arquivo CSV com o prefixo desejado
            csv_filename = os.path.join(os.path.dirname(out_full_path), f"3_CoordMovimento_{base_name}.csv")
            csv_file = open(csv_filename, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['timestamp', 'frame', 'x1', 'y1', 'x2', 'y2', 'confidence'])

            frame_count = 0
            recording = True
            start_time = time.time()
            stop_event.clear()

            # Function to create CSV files
            def create_csv(filename, headers):
                with open(filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)

            # Construct the filenames with the desired structure
            processing_area_filename = os.path.join(folder_path, f"1_ProcessingArea_{base_name}.csv")
            areas_of_interest_filename = os.path.join(folder_path, f"2_AreasOfInterest_{base_name}.csv")


            # Create CSV files for coordinates with the new filenames
            create_csv(processing_area_filename, ['x', 'y'])
            create_csv(areas_of_interest_filename, ['area', 'x1', 'y1', 'x2', 'y2'])

            # Save the coordinates of the processing area (polygon)
            with open(processing_area_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for point in polygon:
                    writer.writerow(point)

            # Save the coordinates of the areas of interest (squares)
            with open(areas_of_interest_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                for i, ((x1, y1), (x2, y2)) in enumerate(squares):
                    writer.writerow([f'Area {i+1}', x1, y1, x2, y2])

            print(f"Files '{processing_area_filename}' and '{areas_of_interest_filename}' have been created.")

             # Iniciar gravação
            frame_manager.start_recording()
            setup_logging(new_folder_path)  # Iniciar logging para esta sessão
            
            recording = True
            stop_event.clear()

            # Iniciar thread de gravação de vídeo
            threading.Thread(target=video_recording_thread).start()

            logging.info(f"Iniciada nova sessão de gravação: Grupo={group_var.get()}, Cobaia={cobaia_number}")

        Button(group_selection_window, text="Confirmar", command=confirm_group).pack()



# Function to stop recording
def stop_recording_button():
    global recording, out, csv_file, csv_writer, capturing, processing
    if recording:
        recording = False
        capturing = False
        processing = False
        
        video_path = out.get(cv2.CAP_PROP_POS_FRAMES)
        csv_path = csv_file.name
        
        if out is not None:
            out.release()
            out = None
        if csv_file is not None:
            csv_file.close()
            csv_file = None
        
        # Verificar integridade após finalizar gravação
        integrity_results = verify_frame_integrity(video_path, csv_path)
        if integrity_results:
            message = (f"Gravação finalizada.\n"
                      f"Frames gravados: {integrity_results['total_recorded']}\n"
                      f"Frames processados: {integrity_results['total_processed']}\n"
                      f"Frames não encontrados: {len(integrity_results['missing_frames'])}\n"
                      f"Frames extras: {len(integrity_results['extra_frames'])}")
            
            messagebox.showinfo("Informação", message)
            logging.info(f"Gravação finalizada - {message}")
        
        stop_event.set()
        
        if out is not None:
            out.release()
            out = None
        if csv_file is not None:
            csv_file.close()
            csv_file = None
        
        restart_camera()

def restart_camera():
    """Função para reiniciar a câmera e a thread de exibição após gravação"""
    global cap, capturing, processing, detection_thread

    # Reabre a captura de vídeo se foi liberada
    if not cap.isOpened():
        cap.open(camera_index)

    # Reinicia a captura e processamento para exibição em tempo real
    capturing = False
    processing = True  # Permitindo a captura para visualização ao vivo
    stop_event.clear()  # Limpar sinal para iniciar nova detecção

    # Inicia uma nova thread de detecção se não houver uma ativa
    start_detection_thread()




# Function to end the project
def end_project_button():
    global recording, group_names, project_folder_path, out, csv_file, csv_writer
    if recording:
        stop_recording_button()
    # Resetar variáveis relacionadas ao projeto
    group_names = []
    project_folder_path = None
    # Opcionalmente, resetar outras variáveis se necessário
    # Informar o usuário
    messagebox.showinfo("Informação", "Projeto encerrado. As configurações foram reiniciadas.")
    # O programa permanece em execução, pronto para iniciar um novo projeto


# Function to end the program
def end_program_button():
    global stop_event, recording, cap
    stop_event.set()
    if recording:
        stop_recording_button()
    # Liberar recursos
    cap.release()
    arduino.close()
    cv2.destroyAllWindows()
    # Encerrar a interface Tkinter
    root.quit()
    # Sair do programa
    import sys
    sys.exit()

def verify_frame_integrity(video_path, csv_path):
    processed_frames = set()
    recorded_frames = set()
    
    logging.info(f"Iniciando verificação de integridade: {os.path.basename(video_path)}")
    
    try:
        # Verificar frames do vídeo
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            recorded_frames.add(frame_count)
            frame_count += 1
        cap.release()
        
        # Verificar registros do CSV
        with open(csv_path, 'r') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Pular cabeçalho
            for row in csv_reader:
                processed_frames.add(int(row[1]))  # coluna frame_count
        
        # Calcular estatísticas
        results = {
            'total_recorded': len(recorded_frames),
            'total_processed': len(processed_frames),
            'missing_frames': processed_frames - recorded_frames,
            'extra_frames': recorded_frames - processed_frames
        }
        
        logging.info(f"Resultados da verificação de integridade: {results}")
        return results
        
    except Exception as e:
        logging.error(f"Erro na verificação de integridade: {e}")
        return None

# Iniciar threads
threading.Thread(target=frame_capture_thread, daemon=True).start()
detection_thread = threading.Thread(target=object_detection_thread)
detection_thread.start()

# Create buttons
def create_buttons():
    create_project_btn = Button(root, text="Criar Projeto", command=create_project_button)
    create_project_btn.pack(side="left")

    start_recording_btn = Button(root, text="Iniciar Gravação", command=start_recording_button)
    start_recording_btn.pack(side="left")

    stop_recording_btn = Button(root, text="Parar Gravação", command=stop_recording_button)
    stop_recording_btn.pack(side="left")

    end_project_btn = Button(root, text="Terminar Projeto", command=end_project_button)
    end_project_btn.pack(side="left")

    end_program_btn = Button(root, text="Terminar Programa", command=end_program_button)
    end_program_btn.pack(side="left")


create_buttons()



# Start the update loop
root.mainloop()
