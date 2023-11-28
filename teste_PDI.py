import cv2
import numpy as np

def comparar_objetos(imagem1, imagem2):
    # Carregar as imagens
    img1 = cv2.imread(imagem1, 0)  # Carregar em escala de cinza
    img2 = cv2.imread(imagem2, 0)  # Carregar em escala de cinza

    # Inicializar o detector de características ORB
    orb = cv2.ORB_create()

    # Encontrar os pontos chave e descritores com ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Inicializar o BFMatcher (Brute-Force Matcher) com a distância de Hamming
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match dos descritores
    matches = bf.match(des1, des2)

    # Ordenar os matches com base nas distâncias
    matches = sorted(matches, key=lambda x: x.distance)

    # Desenhar os matches em uma nova imagem
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Exibir a imagem
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Substitua 'caminho/para/imagem1.jpg' e 'caminho/para/imagem2.jpg' pelos caminhos das suas imagens
comparar_objetos('C:/Users/mateu/Downloads/teste.jpg', 'C:/Users/mateu/Downloads/teste2.jpg')
