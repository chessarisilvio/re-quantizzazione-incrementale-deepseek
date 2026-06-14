# Incremental Requantization for GGUF Models

This tool allows you to requantize specific layers of a GGUF model to Q4_XS while keeping the rest of the model unchanged.

## Descrizione
Il progetto fornisce uno script Python che esegue la riquantizzazione incrementale di layer selezionati di un modello GGUF, convertendoli in formato Q4_XS senza alterare gli altri tensori. È utile per ottimizzare l'uso della VRAM su GPU con memoria limitata (es. Tesla P40) mantenendo la maggior parte del modello nella precisione originale.

## Architettura
- `requantize_incremental.py`: script principale che legge il modello GGUF, processa i layer specificati e scrive il modello di output.
- `verify_incremental.sh`: script di verifica che confronta il modello originale e quello riquantizzato per assicurarsi che i layer non specificati siano identici.
- `layers.json`: esempio di file di configurazione che elenca i nomi dei tensor da riquantizzare.
- Il progetto utilizza la libreria `gguf` per l'accesso diretto ai file GGUF senza dipendere da llama.cpp.

## Installazione
1. **Clona il repository** (se non già fatto):
   ```bash
   git clone <repository-url>
   cd re-quantizzazione-incrementale-deepseek
   ```

2. **Installa le dipendenze Python**:
   ```bash
   pip install --user gguf numpy
   ```
   > Nota: Usa il flag `--user` per evitare conflitti con pacchetti di sistema in ambienti gestiti esternamente.

3. **Verifica la configurazione GPU**:
   Assicurati che la Tesla P40 sia visibile come dispositivo CUDA 1. Puoi impostare i dispositivi visibili tramite variabile d'ambiente:
   ```bash
   export CUDA_VISIBLE_DEVICES=1
   ```
   Aggiungi la riga sopra al tuo profilo di shell (`~/.bashrc`, `~/.zshrc`, ecc.) per persistenza.

4. **Controlla la disponibilità della GPU** prima di eseguire:
   ```bash
   nvidia-smi
   ```
   Cerca la Tesla P40 nell'elenco e conferma che non sia utilizzata da altri processi (o che tu abbia memoria libera sufficiente).

## Uso
Prepara un file JSON che elenchi i nomi dei tensor (layer) che desideri riquantizzare. Il JSON può essere:
- Una semplice lista di stringhe: `["tensor1", "tensor2", ...]`
- Oppure un oggetto con una chiave `"layers"`: `{"layers": ["tensor1", "tensor2", ...]}`

Esegui lo script:

```bash
python requantize_incremental.py \
    --model ./path/to/input/model.gguf \
    --layers-json ./path/to/layers.json \
    --output ./path/to/output/model.gguf   # opzionale
```

Se `--output` è omesso, lo script creerà un file chiamato `input.inc.q4_xs.gguf` nella stessa directory del modello di input.

## Esempi
```bash
export CUDA_VISIBLE_DEVICES=1
nvidia-smi  # verifica che la GPU sia libera
python requantize_incremental.py \
    --model ./models/original-model.gguf \
    --layers-json ./config/layers-to-requantize.json \
    --output ./models/requantized-model.gguf
```

## Stato
✅ COMPLETATO — 2026-06-12
Tutte le fasi sono state completate:
- Identificazione layer da riquantizzare
- Script di riquantizzazione incrementale
- Guida di installazione e uso
- Verifica di integrità post‑riquantizzazione

## Licenza
Questo progetto è rilasciato sotto licenza MIT - vedere il file LICENSE per i dettagli.