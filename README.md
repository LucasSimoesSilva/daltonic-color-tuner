# Sumário

Para salvar a imagem, o OpenCV **exige** valores inteiros entre **0 e 255**.
`uint8` significa **Unsigned Integer de 8 bits** → intervalo **0 a 255**.
Lembrando: o OpenCV usa **BGR** (azul, verde, vermelho), não RGB.  Por isso convertemos **RGB → BGR** antes de salvar.


Em imagens digitais, os valores RGB geralmente não estão em forma linear. Eles passam por uma curva chamada *gama*, que ajusta como o brilho é representado para se aproximar da sensibilidade do olho humano.  
Esse espaço é chamado de *sRGB*. Ele  aplica uma curva fixa (gama = ~2.2) para otimizar a visualização e o uso de bits.

O sRGB é o padrão de monitores e arquivos de imagem. Já o RGB linear é utilizado quando precisamos calcular ou misturar luz de forma correta, como em simulações e efeitos.

A **severidade** representa o **quanto queremos simular** da deficiência visual.

A **Matriz da Deficiência** é uma matriz 3×3 estudada e definida por Machado et al., usada para **simular como uma pessoa com daltonismo percebe as cores**, misturando os canais RGB.

Multiplicar a matriz $M_{def}$ por um vetor de cor $[R,G,B]^T$ (em **RGB linear**) gera uma nova cor que representa essa percepção.

A interpolação:$$M(s) = (1 - s)\,I + s\,M_{def}$$

Permite ajustar o **grau** da simulação:

- s=0: visão normal
- s=1: deficiência severa

O fluxo para ajustar a cor na visão de alguém daltônico seria:

1. sRGB → linear
2. M(s)⋅cor
3. linear → sRGB