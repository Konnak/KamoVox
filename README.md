<p align="center">
  <img src="logo.png" width="120" alt="Logotipo do KamoVox" />
</p>

<h1 align="center">KamoVox</h1>

<p align="center">
  Trocador de voz para Windows.<br />
  A sua voz entra no microfone. A voz que você escolheu sai no Discord, nos jogos e em qualquer chamada.
</p>

<p align="center">
  <a href="https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.1_x64-setup.exe"><img src="https://img.shields.io/badge/Baixar-Windows%20x64-0A84FF?style=for-the-badge" alt="Baixar o instalador para Windows" /></a>
  &nbsp;
  <img src="https://img.shields.io/github/v/release/Konnak/KamoVox?style=for-the-badge&color=64D2FF&label=vers%C3%A3o" alt="Versão atual" />
  &nbsp;
  <img src="https://img.shields.io/badge/Windows-10%20e%2011-1e1e24?style=for-the-badge" alt="Windows 10 e 11" />
</p>

<p align="center">
  Desenvolvido por <a href="https://github.com/Konnak"><b>Konnak</b></a>
</p>

> [!IMPORTANT]
> Baixe **somente** o instalador `KamoVox_0.1.1_x64-setup.exe`.
> Os arquivos `.zip` não são o programa. São pacotes de voz que o KamoVox baixa sozinho, quando você escolhe uma voz. Eles ficam numa [release separada](https://github.com/Konnak/KamoVox/releases/tag/voices), de propósito, para não se misturar com o instalador.

## Baixar

A versão atual é a **0.1.1**, para Windows 10 e 11, 64 bits.

| O que é | Arquivo | Para quem |
|---|---|---|
| O programa | [`KamoVox_0.1.1_x64-setup.exe`](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.1_x64-setup.exe) | Todo mundo. É este o download. |
| Pacotes de voz | [release `voices`](https://github.com/Konnak/KamoVox/releases/tag/voices) | Só o próprio KamoVox. Não baixe na mão. |

O instalador tem cerca de 160 MB, porque algumas vozes já vão dentro dele. As outras são baixadas dentro do programa. Se a página mostrar vários `.zip` soltos, você está na release errada. Volte para a [última versão](https://github.com/Konnak/KamoVox/releases/latest).

## Por que o Windows avisa

O instalador ainda **não tem assinatura digital**. O Windows mostra uma tela azul, “O Windows protegeu o seu PC”, com o texto de publicador desconhecido. Isso não é um antivírus dizendo que encontrou um vírus. É o SmartScreen avisando que o arquivo é novo e que ainda não existe um certificado no nome de quem publicou.

Para abrir:

1. Clique em **Mais informações**.
2. Clique em **Executar assim mesmo**.

O Windows pede permissão de administrador **uma vez**. Essa permissão instala o filtro de áudio no microfone, para o Discord e os outros programas ouvirem a voz alterada. O KamoVox não fica escondido: ao fechar a janela, ele continua na bandeja, ao lado do relógio, e pode ser aberto de novo por ali.

Na análise do instalador da versão 0.1.0 no [VirusTotal](https://www.virustotal.com/gui/file/6cbf82c7e517863b5ed4bac7f629da9b88db52d94465abf7dde3cbc40aeeb975), **2 de 69** motores marcaram o arquivo: DeepInstinct e SecureAge. Os dois usam aprendizado de máquina e costumam marcar instalador novo, sem assinatura. A maioria dos motores dessa análise não marcou. A versão 0.1.1 é outro arquivo, então esse número pode mudar. O aviso azul do Windows continua até o programa ser assinado. Não existe, hoje, um certificado grátis que apague essa tela para todo mundo.

## O que o KamoVox faz

- Troca a voz em tempo real, com vozes prontas e ajuste de tom e timbre.
- Usa voz de inteligência artificial quando o modelo está instalado.
- Coloca efeitos por cima da voz.
- Manda a voz alterada para os outros programas pelo microfone do Windows.
- Deixa você ouvir o resultado no fone antes de soltar no Discord.
- Tem push-to-talk: segure a tecla para falar e solte para silenciar.
- Toca um trecho de música direto no microfone, com atalho.
- Guarda atalhos, perfis e a opção de abrir junto com o Windows.
- Procura atualização no rodapé do próprio programa.

## Instalar

1. Baixe [`KamoVox_0.1.1_x64-setup.exe`](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.1_x64-setup.exe).
2. Se o Windows mostrar a tela azul, use **Mais informações** e **Executar assim mesmo**.
3. Conclua a instalação. Aceite o pedido de administrador quando ele aparecer.
4. Abra o KamoVox, escolha o microfone e ligue o uso nos outros programas.
5. No Discord ou no jogo, selecione esse mesmo microfone.

As vozes grandes são baixadas dentro do programa, na hora em que você escolhe uma. Não é preciso pegar os `.zip` nesta página.

## Requisitos

- Windows 10 ou Windows 11, 64 bits
- Um microfone
- Internet só para baixar atualizações e pacotes de voz

## Atualizar

No rodapé do KamoVox, clique em **Verificar atualizações**.

Se existir uma versão mais nova, o programa baixa o instalador e abre a instalação. O número da versão também abre as notas daquela versão.

## Pacotes de voz

O KamoVox guarda os pacotes de voz numa release própria, chamada [`voices`](https://github.com/Konnak/KamoVox/releases/tag/voices). O programa baixa o arquivo certo quando a voz é escolhida. Essa lista existe para o programa funcionar, não como download para a pessoa.

## Autor

KamoVox foi criado por **Konnak**.

[github.com/Konnak](https://github.com/Konnak)
