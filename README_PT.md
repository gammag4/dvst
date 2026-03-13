# LVSM - Large View Synthesis Model

| [English](README.md) | Português |

Essa é a minha implementação do modelo [LVSM](https://haian-jin.github.io/projects/LVSM/), junto com o código necessário para treiná-lo usando os datasets originais.

Esse é um modelo de Síntese de visão nova, onde dado um conjunto de imagens de uma cena 3D com as respectivas propriedades/poses das câmeras,
o modelo busca gerar uma visão na nova cena, dadas também as propriedades e pose da câmera da visão-alvo.

Esse modelo tem duas coisas em especial quando comparado com outros:

- Ele é generalizável, o que significa que enqunto modelos mais velhos precisam ser retreinados pra cada cena nova, esse pode ser usado em novas cenas sem precisar ser retreinado;
- Ele minimiza viés indutivo usando apenas um vision transformer depois de combinar as visões das câmeras com suas respectivas poses e quebrar tudo em patches. Por iso quando ele é treinado com imagens de alta-resolução, os resultados são bem melhores visualmente que outros modelos.

### Resultados de treino

Por ser um modelo pesado e difícil de treinar, tivemos que o reduzir bastante para usar a resolução de apenas 32x56.

Alguns resultados depois de treinar em uma RTX 4060 Ti com 8GB vRAM por 1 semana:

Rotações nos eixos x, y e z:

<https://github.com/user-attachments/assets/ce782296-1df1-4b93-a4f5-75ea909f551a>

<https://github.com/user-attachments/assets/a9660bc7-d3b8-452f-820d-f88930663891>

<https://github.com/user-attachments/assets/e484d603-0002-4a19-b97d-7b058353156e>

Translações nos eixos x, y e z:

<https://github.com/user-attachments/assets/fff6cbca-1d8c-433d-a742-2a53dbfcff32>

<https://github.com/user-attachments/assets/1ac25592-2af5-4f4d-a909-674fd8e9c493>

<https://github.com/user-attachments/assets/2cd17c85-b676-4dfd-be81-d4385bb02190>

Observação:
Quando treinando com recursos restritos, deve-se aumentar bastante os valores dos betas.
Isso cria um efeito de média similar ao que GrokFast faz.
Sem isso, o modelo não converge.

### NN Image

Para facilitar configurar o ambiente, nós também construímos e usamos uma imagem docker dada [aqui](https://github.com/gammag4/nn-image).

### DVST

Esse modelo era originalmente uma tentativa de extender o LVSM para cenas dinâmicas,
chamado Dynamic View Synthesis Transformer (DVST),
mas por falta de recursos, o autor parou de trabalhar nele no momento.
Ele já tem partes do código para treinar com cenas dinâmicas e também para treinar com várias GPUs usando DDP, mas algumas partes não estão completas.
