import cv2
import mediapipe as mp

# Inicializa as ferramentas de IA do MediaPipe para Python
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicia a captura da webcam do computador (0 = câmera padrão)
cap = cv2.VideoCapture(0)

print("=== GREMORY PY-MATRIX INICIADO ===")
print("Pressione 'ESC' na janela do vídeo para fechar.")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Inverte o vídeo para efeito espelho e converte para RGB
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Desenha o contorno do rosto em Vermelho (Gremory Style)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Mapeia os pontos dos lábios
                lip_top = face_landmarks.landmark[13].y
                lip_left = face_landmarks.landmark[61].y
                lip_right = face_landmarks.landmark[291].y
                
                smile_score = ((lip_left + lip_right) / 2) - lip_top
                smile_score *= 100
                
                emocao = "NEUTRO"
                cor_texto = (255, 255, 255) # Branco
                
                if smile_score < 1.9:
                    emocao = "FELIZ 😄"
                    cor_texto = (0, 255, 0) # Verde
                elif smile_score > 3.7:
                    emocao = "TRISTE 😢"
                    cor_texto = (255, 0, 0) # Azul
                
                cv2.putText(image, f"EMOCAO: {emocao}", (30, 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, cor_texto, 2, cv2.LINE_AA)

        # Abre a janela com o sistema rodando
        cv2.imshow('Gremory Co. - Py-Matrix Engine', image)
        
        # Fecha se apertar a tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
