<div align="center">
  <img src="figures/cabecalho_transp.png" alt="Cabeçalho" width="100%">
</div>

# A otimização de hiperparâmetros aplicada para a modelagem do índice de refração 

> Este é um projeto da disciplina de Aprendizado de Máquina, do segundo semestre do Bacharelado em Ciência e Tecnologia, da Ilum Escola de Ciência.


Para este trabalho, visou-se utilizar o método bayesiano de otimização Optuna para performar a otimização de  hiperparâmetros de modelos de aprendizado de máquina lineares e não lineares para a predição do índice de refração de materiais inorgânicos.

O principal material deste repositório é o notebook **refractive_predict.ipynb**, em que é apresentado o problema de modelagem, a explicação da abordagem para a indução dos modelos e da otimização de seus hiperparâmetros. Ao fim, são discutidos os resultados obitidos a partir da otimização com o Optuna.

## 📖 Conteúdo

<ul>
  <li> :file_folder: <b>Projeto</b>: contém os scripts, arquivos e dados utilizados no projeto.</li>
    <ul>
      <li> :file_folder: <b>figures</b>: contém as imagens de cabeçalho e rodapé utilizadas na documentação do projeto.</li>
        <ul>
          <li> :framed_picture: <b>cabecalho.png</b>: imagem utilizada no cabeçalho do projeto.</li>
          <li> :framed_picture: <b>cabecalho)transp.png</b>: imagem utilizada no cabeçalho do README.</li>          
          <li> :framed_picture: <b>rodape.png</b>: imagem utilizada no rodapé do projeto.</li>
        </ul>
      <li> :file_folder: <b>hpc_files</b>: contém os arquivos Python (<code>.py</code>) e scripts shell (<code>.sh</code>) para a otimização de hiperparâmetros em ambiente HPC.</li>
      <li> :file_folder: <b>optuna_files</b>: armazena todos os arquivos de banco de dados SQLite (<code>.db</code>) gerados pelos estudos de otimização do Optuna.</li>
      <li> :card_file_box: <b>dielectric_constant.pkl</b>: conjunto de dados principal contendo os dados dos materiais e descritores utilizados ao longo do projeto.</li>
      <li> :page_facing_up: <b>refractive_predict.ipynb</b>: notebook Jupyter principal contendo toda a metodologia, análise e discussão dos resultados do projeto.</li>
      <li> :page_facing_up: <b>LICENSE</b>: licença de uso e distribuição do projeto.</li>
      <li> :page_facing_up: <b>README.md</b>: documentação principal do projeto.</li>
      <li> :spiral_notepad: <b>requirements.txt</b>: arquivo contendo as bibliotecas e versões necessárias para executar o projeto.</li>
    </ul>
</ul>


## 📦 Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/cg-oc/refractive_predict.git
cd refractive_predict
pip install -r requirements.txt
```

## 🐍 Versão do Python e módulos

- Python: 3.13.7
- matminer==0.10.1
- matplotlib==3.11.1
- numpy==2.5.2
- optuna==4.5.0
- pandas==2.3.3
- scikit-learn==1.9.0
- seaborn==0.13.2


## 📄 Licença

GNU General Public License v3.0

Leia o arquivo `LICENSE` para mais detalhes.


# Docente

<table>
  <td align="center">
      <a href="https://github.com/drcassar">
        <img src="https://avatars.githubusercontent.com/u/9871905?v=4" width="100px;" alt="Foto do docente Daniel Cassar no Github"/><br>
        <b>Prof. Dr. Daniel R. Cassar</b>
      </a>
    </td>
</table>


# Autor

<table>
  <td align="center">
      <a href="https://github.com/cg-oc">
        <img src="https://avatars.githubusercontent.com/u/166063037?v=4" width="100px;" alt="Foto do autor Carlos Gabriel"/><br>
        <b>Carlos Gabriel</b>
      </a>
    </td>
  <td></table>
