import cv2
import mediapipe as mp
import math

# Inicializa as ferramentas de IA do MediaPipe para Python
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Inicia a captura da webcam do computador (0 = câmera padrão)
cap = cv2.VideoCapture(0)

print("=== GREMORY PY-MATRIX MATRIX-PRO INICIADO ===")
print("Pressione 'ESC' na janela do vídeo para fechar.")

# Função auxiliar para calcular a distância 3D entre dois pontos do rosto
def calcular_distancia(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

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
                # Desenha a grade de pontos em Vermelho (Gremory Style)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_FACE_OVAL,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # --- CAPTURA DE PONTOS CHAVE ---
                landmarks = face_landmarks.landmark
                
                # Boca (Sorriso e Abertura)
                canto_boca_esq = landmarks[61]
                canto_boca_dir = landmarks[291]
                labio_superior = landmarks[13]
                labio_inferior = landmarks[14]
                
                # Sobrancelhas (Distância entre elas e altura)
                sobrancelha_esq_interna = landmarks[55]
                sobrancelha_dir_interna = landmarks[285]
                nariz_topo = landmarks[6]
                
                # Olhos (Abertura para surpresa ou piscar)
                olho_sup = landmarks[159]
                olho_inf = landmarks[145]

                # --- CÁLCULO DAS MÉTRICAS ---
                largura_boca = calcular_distancia(canto_boca_esq, canto_boca_dir)
                abertura_boca = calcular_distancia(labio_superior, labio_inferior)
                distancia_sobrancelhas = calcular_distancia(sobrancelha_esq_interna, sobrancelha_dir_interna)
                abertura_olho = calcular_distancia(olho_sup, olho_inf)
                
                # Altura média dos cantos da boca em relação ao centro do lábio (Sorriso)
                altura_cantos_boca = (canto_boca_esq.y + canto_boca_dir.y) / 2
                smile_score = altura_cantos_boca - labio_superior.y
                smile_score *= 100

                # --- LÓGICA DO DETECTOR DE EMOÇÕES MULTI-ALVO ---
                emocao = "NEUTRO"
                cor_texto = (255, 255, 255)  # Branco BGR

                # 1. PISCANDO / OLHO FECHADO
                if abertura_olho < 0.015:
                    emocao = "PISCANDO / DORMINDO 😴"
                    cor_texto = (0, 255, 255)  # Amarelo

                # 2. SURPRESO (Olhos muito abertos e boca aberta verticalmente)
                elif abertura_olho > 0.032 and abertura_boca > 0.04:
                    emocao = "SURPRESO 😲"
                    cor_texto = (255, 255, 0)  # Ciano

                # 3. FELIZ (Cantos da boca curvados para cima)
                elif smile_score < 1.9:
                    emocao = "FELIZ 😄"
                    cor_texto = (0, 255, 0)  # Verde

                # 4. BRAVO / IRADO (Sobrancelhas juntas e baixas)
                elif distancia_sobrancelhas < 0.042:
                    emocao = "BRAVO 😡"
                    cor_texto = (0, 0, 255)  # Vermelho

                # 5. TRISTE (Cantos da boca caídos)
                elif smile_score > 3.7:
                    emocao = "TRISTE 😢"
                    cor_texto = (255, 0, 0)  # Azul

                # Desenha o texto da emoção na tela
                cv2.putText(image, f"SISTEMA: {emocao}", (30, 50), 
                            cv2.FONT_HERSHEY_COMPLEX, 1, cor_texto, 2, cv2.LINE_AA)
                
                # Exibe dados técnicos no canto para parecer Matrix profissional
                cv2.putText(image, f"Smile Score: {round(smile_score, 2)}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Abre a janela com o sistema rodando
        cv2.imshow('Gremory Co. - Py-Matrix Pro Engine', image)
        
        # Fecha se apertar a tecla ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
