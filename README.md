<p align="center">
  <img src="logo.png" width="96" alt="KamoVox" />
</p>

<h1 align="center">KamoVox</h1>

<p align="center">
  Trocador de voz em tempo real para Windows. Motor em Rust, interface Tauri, inferência ONNX local.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000?style=flat-square&logo=rust&logoColor=white" alt="Rust" />
  <img src="https://img.shields.io/badge/Tauri_2-24C8DB?style=flat-square&logo=tauri&logoColor=white" alt="Tauri 2" />
  <img src="https://img.shields.io/badge/Windows_10_/_11-0078D6?style=flat-square&logo=windows&logoColor=white" alt="Windows 10 e 11" />
  <img src="https://img.shields.io/badge/gratuito-30d158?style=flat-square" alt="Gratuito" />
  <img src="https://img.shields.io/github/v/release/Konnak/KamoVox?style=flat-square&label=release" alt="Release" />
</p>

<p align="center">
  <a href="https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe"><img src="https://img.shields.io/badge/Windows_x64-0A84FF?style=for-the-badge&logo=windows&logoColor=white" alt="Baixar o instalador Windows x64" /></a>
</p>

<p align="center">
  <img src="docs/hero.png" width="920" alt="KamoVox" />
</p>

<p align="center">
  <a href="#recursos">Recursos</a> ·
  <a href="#interface">Interface</a> ·
  <a href="#arquitetura">Arquitetura</a> ·
  <a href="#instalação">Instalação</a> ·
  <a href="#compilar">Compilar</a>
</p>

## Recursos

- **DSP em tempo real.** Pitch, formante, textura, reverb e vocoder no crate `kamovox-dsp`. Latência baixa, para uso junto com jogo e Discord.
- **Inferência local.** Conversão RVC em ONNX (`ort`, DirectML ou CPU). Presets de bloco: 160 ms, 250 ms ou 400 ms.
- **APO de microfone.** `kamovox-apo` registra um dispositivo no Windows. Discord e jogos leem esse dispositivo.
- **Monitor.** Saída no fone, independente do dispositivo virtual.
- **Atalhos globais.** Push-to-talk, troca de voz, disparo de faixa no microfone (`symphonia`).
- **Perfis.** Voz, efeito e dispositivo persistidos no cliente.

Processamento 100% local. Nenhuma amostra de áudio é enviada para servidores externos.

## Interface

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/vozes.png" alt="Vozes clássicas" />
      <sub>Vozes clássicas, pitch e formante.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="docs/voz-ia.png" alt="Voz IA" />
      <sub>Modelos RVC instalados. Calibração de pitch por voz.</sub>
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/efeitos.png" alt="Efeitos" />
      <sub>Efeitos e vocoder.</sub>
    </td>
    <td width="50%" valign="top">
      <img src="docs/ajustes.png" alt="Áudio" />
      <sub>Dispositivos, monitor e roteamento do APO.</sub>
    </td>
  </tr>
  <tr>
    <td colspan="2" valign="top">
      <img src="docs/atalhos.png" alt="Atalhos" />
      <sub>Atalhos globais. Letras exigem modificador. F1–F12 podem ir sozinhas.</sub>
    </td>
  </tr>
</table>

## Arquitetura

| Crate | Função |
| --- | --- |
| `kamovox-dsp` | Pitch, formante, efeitos, vocoder. FFT via `realfft`. |
| `kamovox-ai` | ONNX Runtime (`ort`): ContentVec, RMVPE, modelo RVC. DirectML ou CPU. |
| `kamovox-apo` | Audio Processing Object. Expõe o microfone virtual. |
| `kamovox-link` | Canal entre o processo do app e o APO. |
| UI | Tauri 2, TypeScript, Vite. |

Captura com `cpal`. Fila de áudio em `ringbuf`. Release com LTO e `opt-level = 3`.

Os pacotes de voz ficam na release [`voices`](https://github.com/Konnak/KamoVox/releases/tag/voices). O cliente baixa o zip ao selecionar o modelo. ContentVec e RMVPE são obtidos no primeiro uso da voz IA.

## Instalação

Windows 10 ou 11, x64. O instalador é gratuito.

**[KamoVox_0.1.6_x64-setup.exe](https://github.com/Konnak/KamoVox/releases/latest/download/KamoVox_0.1.6_x64-setup.exe)** (~160 MB)

O binário não tem assinatura Authenticode. O SmartScreen exibe “O Windows protegeu o seu PC”. **Mais informações** → **Executar assim mesmo**. O pedido de administrador na primeira execução registra o APO.

1. Execute o instalador.
2. Em Ajustes, selecione a entrada e ative a voz nos programas.
3. No Discord ou no jogo, selecione o dispositivo do KamoVox.

**Verificar atualizações**, no rodapé, consulta a release e pede confirmação antes de instalar.

## Compilar

Pré-requisitos: Rust stable (MSVC), Node.js, WebView2, Visual Studio Build Tools.

Este repositório publica o instalador e esta página. O código dos crates não está aqui.

Na árvore de trabalho, a partir de `app/`:

```sh
npm install
npm run tauri dev
```

```sh
npm run tauri build
```

O NSIS sai em `src-tauri/target/release/bundle/nsis/`.

---

[Konnak](https://github.com/Konnak)
