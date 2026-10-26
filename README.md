<p align="center">
  <img src="logo.png" width="120" alt="KamoVox" />
</p>

<h1 align="center">KamoVox</h1>

<p align="center">
  Modificador de voz em tempo real para Windows.<br />
  Vozes clássicas, conversão RVC por ONNX, efeitos, vocoder e microfone virtual<br />
  para Discord, jogos e chamadas. Motor de áudio em Rust, interface Tauri. Gratuito.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000?style=flat-square&logo=rust&logoColor=white" alt="Rust" />
  <img src="https://img.shields.io/badge/Tauri_2-24C8DB?style=flat-square&logo=tauri&logoColor=white" alt="Tauri 2" />
  <img src="https://img.shields.io/badge/Windows_10_/_11-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Windows 10 e 11" />
  <img src="https://img.shields.io/badge/gratuito-30d158?style=flat-square" alt="Gratuito" />
  <img src="https://img.shields.io/github/v/release/Konnak/KamoVox?style=flat-square&label=versão" alt="Versão" />
</p>

<p align="center">
  <a href="https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe"><img src="https://img.shields.io/badge/Download_%C2%B7_Windows_x64-0A84FF?style=for-the-badge&logo=windows&logoColor=white" alt="Download · Windows x64" /></a>
</p>

<p align="center">
  <img src="docs/hero.png" width="920" alt="KamoVox na página de vozes clássicas" />
</p>

<p align="center">
  Vozes clássicas, categorias, tom e entonação.<br />
  Versão 0.1.6
</p>

<p align="center">
  <a href="#recursos">Recursos</a> ·
  <a href="#interface">Interface</a> ·
  <a href="#no-programa">No programa</a> ·
  <a href="#arquitetura">Arquitetura</a> ·
  <a href="#instalação">Instalação</a> ·
  <a href="#downloads">Downloads</a> ·
  <a href="#compilar">Compilar</a>
</p>

<table>
  <tr>
    <td width="25%" valign="top">
      <h3>🎛️ Vozes clássicas</h3>
      22 presets em cinco categorias, tom de −24 a +24 semitons, entonação de −12 a +12, textura e 16 personagens. O modo Combinar empilha mais de uma voz. O motor aceita uma cadeia de até 4 presets.
    </td>
    <td width="25%" valign="top">
      <h3>⚡ Motor em Rust</h3>
      Captura com <code>cpal</code>, fila em <code>ringbuf</code>, DSP em <code>kamovox-dsp</code>. A interface é Tauri 2. O release compila com LTO e <code>opt-level = 3</code>. O cartão da barra mostra o estado do áudio e o atraso das vozes clássicas.
    </td>
    <td width="25%" valign="top">
      <h3>🧠 Voz IA</h3>
      Conversão RVC em ONNX (<code>ort</code>), DirectML ou CPU. Detecção de tom RMVPE ou YIN. Blocos de 160 ms, 250 ms ou 400 ms, ou valores manuais. Importa <code>.pth</code>.
    </td>
    <td width="25%" valign="top">
      <h3>🎙️ Microfone virtual</h3>
      <code>kamovox-apo</code> registra um Audio Processing Object no dispositivo do Windows. Discord e jogos selecionam esse dispositivo. O monitor manda a mesma saída para o fone.
    </td>
  </tr>
</table>

## Interface

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/vozes.png" alt="Página de vozes clássicas" />
      <p><sub>Vozes clássicas, tom e entonação.</sub></p>
      <h3>Vozes clássicas</h3>
      <p>Um toque liga o preset. Outro toque volta ao sinal direto. As categorias são Pessoas, Criaturas, Máquinas, Transmissão e Ambiente. O filtro Todas mostra as 22.</p>
      <p><strong>Tom</strong> desloca o pitch em semitons, em passos de 0,1. <strong>Entonação</strong> desloca os formantes sem acompanhar o tom: negativo alarga o corpo da voz, positivo estreita. Em 0, o formante fica no timbre de entrada. <strong>Textura</strong> soma respiração, rouquidão, vibrato e nasal, cada um de 0 a 100%. <strong>Personagens</strong> aplica um conjunto já preenchido desses seis valores. <strong>Zerar</strong> devolve tom, entonação e textura ao zero.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/voz-ia.png" alt="Página de voz IA" />
      <p><sub>Modelos RVC instalados, com calibrar por carta.</sub></p>
      <h3>Conversão RVC</h3>
      <p>Com <strong>Usar voz IA</strong> ligado, o modelo substitui os presets clássicos. Efeitos, textura, equalizador e o roteamento do microfone continuam no caminho. Cada carta tem <strong>Calibrar</strong>: o motor mede o tom da entrada já aberta e grava o resultado para sugerir tom e entonação naquela voz.</p>
      <p>O tom da voz IA vai de −24 a +24 semitons e entra antes da conversão. A entonação, de −12 a +12, entra depois. <strong>Importação</strong> aceita um modelo RVC <code>.pth</code>. A biblioteca baixa o zip da release <a href="https://github.com/Konnak/KamoVox/releases/tag/voices"><code>voices</code></a> quando a voz é escolhida. ContentVec (<code>contentvec_768l12.onnx</code>) e RMVPE (<code>rmvpe.onnx</code>) entram no primeiro uso.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/efeitos.png" alt="Página de efeitos" />
      <p><sub>Efeitos, vocoder e mistura com o sinal original.</sub></p>
      <h3>Efeitos e vocoder</h3>
      <p>Nove efeitos, ligados por cima da voz: Caverna, Salão, Sala, Coro, Fantasma, Eco, Reflexo, Jato e Garrafa. Mais de um pode ficar ativo ao mesmo tempo.</p>
      <p>O vocoder usa o envelope da fala para excitar uma portadora: Sintetizador, Acorde, Sino, Órgão, Coral, Zumbido ou Vento. A intensidade vai de 0 a 100%. <strong>Misturar voz original</strong> soma o sinal de entrada à saída.</p>
    </td>
    <td width="50%" valign="top">
      <img src="docs/ajustes.png" alt="Página de ajustes" />
      <p><sub>Entrada, monitor, APO e calibração.</sub></p>
      <h3>Áudio e roteamento</h3>
      <p>A entrada é o microfone de captura. A saída do monitor é o fone, separada do dispositivo que os outros programas leem. <strong>Calibração automática</strong> usa o mesmo fluxo do motor, não o seletor de microfone do sistema fora do app. O resultado em hertz fica salvo e serve de base para outras vozes.</p>
      <p><strong>Voz alterada nos programas</strong> liga o APO no endpoint escolhido. Instalar ou trocar o endpoint pede administrador e reinicia o áudio do Windows por alguns segundos. O cabeçalho mostra a taxa em que o dispositivo está rodando. O equalizador tem graves e agudos em dB, além de ganho, sensibilidade (noise gate) e redução de ruído.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/atalhos.png" alt="Página de atalhos" />
      <p><sub>Atalhos globais, vozes, efeitos e faixas.</sub></p>
      <h3>Atalhos</h3>
      <p>Atalhos globais para ligar o modificador, ligar a voz IA, ligar a voz nos programas e ligar o monitor. As listas de vozes clássicas, vozes IA, efeitos e faixas abrem um atalho por item. <strong>Personalizado</strong> cria um atalho extra.</p>
      <p>Letras exigem Ctrl, Alt ou Shift. F1 a F12 podem ir sozinhas. O push-to-talk fica em Ajustes, na aba própria: segurar fala, ou o modo invertido, em que segurar muta.</p>
    </td>
    <td width="50%" valign="top">
      <h3>Música e perfis</h3>
      <p>Em <strong>Música</strong>, uma faixa é um arquivo local. O atalho dela escreve o áudio no microfone virtual, no mesmo caminho da voz. Formatos lidos pelo <code>symphonia</code>: mp3, wav, ogg, flac, m4a, aac e mp4.</p>
      <p><strong>Perfis</strong> gravam o conjunto atual de parâmetros: voz, efeitos, vocoder, textura e ajustes ligados a esse estado. Aplicar um perfil troca esse conjunto. Excluir o perfil não altera a voz que está soando.</p>
      <p>Fechar a janela deixa o processo na bandeja, ao lado do relógio. Sair é pelo menu da bandeja. <strong>Iniciar com o Windows</strong> registra a abertura no login. O tema segue o sistema, ou fica claro, ou escuro.</p>
      <p><strong>Verificar atualizações</strong>, no rodapé, consulta a última release. Se houver versão nova, o cliente pergunta antes de baixar o instalador e abrir a atualização.</p>
    </td>
  </tr>
</table>

## No programa

<table>
  <tr>
    <td width="50%" valign="top">
      <h3>🎛️ 22 vozes clássicas</h3>
      <p><strong>Pessoas.</strong> Masculina · Feminina · Criança · Gigante · Hélio</p>
      <p><strong>Criaturas.</strong> Demônio · Monstro · Alienígena · Mutação · Mutação lenta · Mutação rápida</p>
      <p><strong>Máquinas.</strong> Robô · Ciborgue · 8-bit · Clone</p>
      <p><strong>Transmissão.</strong> Rádio · Telefone · Walkie-talkie · Megafone</p>
      <p><strong>Ambiente.</strong> Subaquático · Sussurro · Silêncio</p>
    </td>
    <td width="50%" valign="top">
      <h3>🎭 16 personagens</h3>
      <p>Homem · Mulher · Jovem · Criança · Idoso · Narrador · Vilão · Gigante · Anão · Orc · Bruxa · Fada · Pato · Máscara sombria · Sussurro · Cantor</p>
      <p>Cada um é um ponto fixo de tom, entonação, respiração, rouquidão, vibrato e nasal. O cartão ativo é o que coincide com os sliders.</p>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <h3>✨ Efeitos e vocoder</h3>
      <p>Caverna · Salão · Sala · Coro · Fantasma · Eco · Reflexo · Jato · Garrafa</p>
      <p>Portadoras: Sintetizador · Acorde · Sino · Órgão · Coral · Zumbido · Vento</p>
      <p>Mistura com o sinal original, em paralelo.</p>
    </td>
    <td width="50%" valign="top">
      <h3>🧠 Inferência</h3>
      <p>Presets: Desempenho (bloco 160 ms, contexto 300 ms, transição 40 ms), Equilibrado (250 / 600 / 60), Qualidade (400 / 1200 / 100), Personalizado.</p>
      <p>No personalizado, o bloco vai de 100 a 800 ms, o contexto de 100 a 2500 ms e a transição de 20 a 150 ms. Dispositivo: placa de vídeo (DirectML) ou processador. Tom: RMVPE ou YIN.</p>
      <p>A biblioteca publicada na release <code>voices</code>: Baby, Bolsonaro, Buzz, Chapolin, Egirl, Elsa, Inês Brasil, Jessie, Patolino, Wendel Bezerra, Woody, Zezão. O cliente também importa <code>.pth</code>.</p>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <h3>🎚️ O que mais entra no caminho do áudio</h3>
      <p>Ganho de entrada · sensibilidade (noise gate) · graves · agudos · redução de ruído · janela de pitch mais longa nas vozes clássicas (menos metálico, um pouco mais de atraso) · push-to-talk com tecla e modo invertido · monitor no fone · atalhos globais · faixas no microfone · perfis · bandeja · iniciar com o Windows · tema sistema, claro ou escuro · verificação de update com confirmação.</p>
    </td>
  </tr>
</table>

## Arquitetura

O caminho do sinal é local: microfone → fila → DSP clássico ou ONNX → APO e monitor.

| Crate | Função |
| --- | --- |
| `kamovox-dsp` | Pitch, formante, textura, efeitos, vocoder. FFT em `realfft`. Noise gate e respiração no mesmo crate. |
| `kamovox-ai` | ONNX Runtime (`ort` 2, API 31, DirectML). ContentVec, RMVPE e o modelo RVC. |
| `kamovox-apo` | Audio Processing Object do Windows. Expõe o dispositivo que os outros programas abrem. |
| `kamovox-link` | Canal entre o processo do app e o APO. |
| UI | Tauri 2, TypeScript, Vite. Estado em `kamovox.settings.v1`. |

- **Captura.** `cpal` 0.15. A taxa exibida em Ajustes é a do dispositivo (48 kHz quando o endpoint entrega essa taxa).
- **Fila.** `ringbuf`. Se a saída atrasa, o bloco mais antigo é descartado em vez de a latência crescer sem limite.
- **Voz IA.** O bloco define o atraso mínimo da conversão. Contexto maior não soma esse atraso; aumenta o áudio anterior analisado. A transição soma ao atraso.
- **Release.** `codegen-units = 1`, LTO, `opt-level = 3`, `panic = abort`, símbolos removidos. Em debug, `kamovox-dsp` e `kamovox-ai` já compilam com `opt-level = 3`.
- **Rede.** O cliente baixa o instalador de atualização, os zips da release `voices` e os dois ONNX de inferência. O áudio da captura não entra nesses pedidos.

## Instalação

Windows 10 ou 11, x64.

1. Baixe [`KamoVox_0.1.6_x64-setup.exe`](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe).
2. Se o SmartScreen abrir, use **Mais informações** e **Executar assim mesmo**. O arquivo não tem assinatura Authenticode.
3. Conclua o instalador. O pedido de administrador registra o APO.
4. Em Ajustes, escolha o microfone, instale o endpoint e ligue **Voz alterada nos programas**.
5. No Discord ou no jogo, selecione esse dispositivo.
6. Ligue o modificador no cartão da barra esquerda e escolha uma voz.

Alguns modelos de voz IA já vão no instalador (~160 MB). Os outros saem da release [`voices`](https://github.com/Konnak/KamoVox/releases/tag/voices), baixados pelo cliente. Esses zips não são o programa.

## Downloads

| Build | Arquivo |
| --- | --- |
| Windows x64 | [`KamoVox_0.1.6_x64-setup.exe`](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe) |

O instalador é o único arquivo para instalar o programa. Notas da versão ficam na [release](https://github.com/Konnak/KamoVox/releases/latest).

| Release | Conteúdo |
| --- | --- |
| [`voices`](https://github.com/Konnak/KamoVox/releases/tag/voices) | Zips de modelo. O cliente baixa o arquivo ao selecionar a voz. |

## Compilar

Pré-requisitos: Rust stable (MSVC), Node.js, WebView2, Visual Studio Build Tools.

Este repositório publica o instalador e esta página. Os crates não estão incluídos.

Na árvore de trabalho, a partir de `app/`:

```sh
npm install
npm run tauri dev
```

```sh
npm run tauri build
```

`tauri build` gera o APO, compila o frontend e o binário. O NSIS sai em `src-tauri/target/release/bundle/nsis/`.

---

[Konnak](https://github.com/Konnak)
