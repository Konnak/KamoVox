<p align="center">
  <img src="logo.png" width="104" alt="Logotipo do KamoVox" />
</p>

<h1 align="center">KamoVox</h1>

<p align="center">
  Trocador de voz para Windows.<br />
  Você fala no microfone. A voz que escolheu sai no Discord, no jogo, na call.
</p>

<p align="center">
  <a href="https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe"><img src="https://img.shields.io/badge/Baixar-Windows%2010%20%2F%2011-0A84FF?style=for-the-badge" alt="Baixar o instalador para Windows" /></a>
</p>

<p align="center">
  <img src="docs/vozes.png" width="920" alt="Tela de vozes prontas do KamoVox" />
</p>

<p align="center">
  Feito por <a href="https://github.com/Konnak">Konnak</a> · sem mensalidade · sem criar conta
</p>

O botão azul da barra esquerda liga o trocador. Enquanto ele estiver aceso, o que os outros ouvem não é a sua voz de verdade — é a voz do programa. Fecha a janela e o camaleão continua na bandeja, ao lado do relógio.

## Vozes prontas

Toque numa carta para ativar. Toque de novo para voltar ao natural. As categorias separam gente, criatura, máquina, rádio e ambiente. Se não quiser ficar girando slider, **Personagens** já traz combinações prontas (vilão, narrador, criança, e por aí).

Embaixo da lista:

- **Tom** — a voz fica mais grossa ou mais fina
- **Entonação** — muda o “corpo” da garganta, sem alterar a nota
- **Textura** — ar na voz, rouquidão, vibrato

O modo **Combinar** mistura duas vozes prontas ao mesmo tempo.

## Voz IA

<p align="center">
  <img src="docs/voz-ia.png" width="920" alt="Biblioteca de vozes de inteligência artificial" />
</p>

Aqui a conversão é outra. A sua fala vira a voz de um modelo, não um efeito em cima do microfone. As vozes baixam **dentro do programa**, quando você pede. Não precisa caçar `.zip` nesta página.

Cada carta tem **Calibrar**: o KamoVox ouve o microfone que já está ligado, mede o tom da sua voz e acerta o deslocamento. Homem numa voz aguda, mulher numa voz grave — isso deixa de ser chute.

Dá para importar um modelo RVC (`.pth`) se você tiver o seu. Placa de vídeo ajuda; se não tiver, o programa usa o processador e avisa o atraso.

Com a voz IA ligada, ela substitui as vozes prontas. Efeito, textura e o microfone continuam valendo.

## Efeitos

<p align="center">
  <img src="docs/efeitos.png" width="920" alt="Página de efeitos, vocoder e mistura" />
</p>

Caverna, salão, eco, coro, vocoder. Entram por cima da voz, não no lugar dela. O vocoder usa o ritmo da sua fala para “tocar” um instrumento — sintetizador, sino, órgão, vento. Se quiser, mistura a voz original junto, para não ficar 100% personagem.

## No Discord e nos jogos

<p align="center">
  <img src="docs/ajustes.png" width="920" alt="Ajustes de microfone e uso em outros programas" />
</p>

O KamoVox instala um microfone no Windows. É **esse** microfone que o Discord, o jogo ou a call têm que usar. Sem essa etapa, o outro lado continua ouvindo a sua voz normal.

Na mesma tela:

- **Calibração automática** — o programa escuta você e acerta o tom
- **Ouvir no fone** — testa o resultado antes de soltar na call. Use fone, senão vira eco
- **Push-to-talk** — segura a tecla para falar, solta para silenciar (ou o contrário)
- Abrir junto com o Windows, tema claro / escuro / sistema

O áudio precisa estar ligado. O cartão **Trocador de voz**, embaixo da barra, mostra se está saindo voz e quanto de atraso tem.

## Atalhos, música, perfis

<p align="center">
  <img src="docs/atalhos.png" width="920" alt="Atalhos de teclado do KamoVox" />
</p>

Uma tecla para ligar o modificador, trocar de voz, mandar um trecho de música no microfone. Letra precisa de Ctrl, Alt ou Shift. F1 a F12 podem ir sozinhas.

Em **Música**, você coloca um áudio e dispara com o atalho — o som vai para o microfone, não só para o seu fone. **Perfis** guardam a combinação inteira (voz, efeito, microfone) para voltar nela depois.

O rodapé tem a versão e o botão de atualizar. Se existir versão nova, o KamoVox pergunta antes de instalar. Não instala sozinho.

## Baixar

O arquivo certo é só este:

**[KamoVox_0.1.6_x64-setup.exe](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe)** — cerca de 160 MB, Windows 10 e 11, 64 bits.

Alguns pacotes de voz já vêm no instalador. O resto o programa baixa quando você escolhe a voz.

Os `.zip` que aparecem no GitHub **não são o programa**. São os pacotes. Ficam numa [release separada](https://github.com/Konnak/KamoVox/releases/tag/voices), de propósito, para não se misturar com o instalador. Não baixe eles na mão.

### A tela azul do Windows

O instalador ainda não tem assinatura digital. O Windows mostra “O Windows protegeu o seu PC”, com publicador desconhecido. Isso é o SmartScreen, não um antivírus dizendo que achou vírus.

Clique em **Mais informações**, depois em **Executar assim mesmo**.

Na primeira abertura o Windows pede administrador **uma vez**: é para instalar o microfone virtual. Fechar a janela não desliga o KamoVox — ele fica na bandeja. Para sair de verdade, use o menu de lá.

## Dois minutos para usar

1. Instale o `.exe`. Se aparecer a tela azul, **Mais informações** → **Executar assim mesmo**.
2. Abra o KamoVox, escolha o microfone, ligue **Voz alterada nos programas**.
3. No Discord (ou no jogo), selecione o microfone do KamoVox.
4. Ligue o botão de energia na barra esquerda e escolha uma voz.

Internet só entra na história para atualizar e para baixar voz nova.

---

KamoVox é um projeto pessoal. O camaleão da barra não cobra café, mas o botão **Apoie o projeto**, dentro do programa, existe se alguém quiser.

[github.com/Konnak](https://github.com/Konnak)
