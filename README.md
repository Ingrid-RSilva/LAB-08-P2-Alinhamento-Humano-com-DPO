# Laboratório 08 — Alinhamento Humano com DPO

Pipeline completo de alinhamento de LLM utilizando **Direct Preference Optimization (DPO)** para garantir comportamento **HHH** — Helpful, Honest, Harmless.

---

## Configuração Necessária

Antes de executar o notebook, substitua a chave da API Groq na célula de **Configuração Global**:

```python
GROQ_API_KEY = "sua_chave_aqui"
```

A chave pode ser obtida gratuitamente em: https://console.groq.com

---

## Estrutura do Repositório

```
lab08-dpo/
├── lab08_dpo_alignment.ipynb   # Notebook principal
├── dpo_dataset.jsonl           # Dataset de preferências gerado (≥30 pares)
├── dpo-aligned-model/          # Modelo alinhado salvo após treinamento
│   ├── experiment_log.json     # Log do experimento (métricas, configs)
│   └── adapter_model.bin       # Pesos LoRA do modelo ator
└── README.md                   # Este arquivo
```

---

## Roteiro de Implementação

### Passo 1 — Dataset de Preferências HHH

O dataset foi gerado via **Groq API (Llama 3.1-8b-instant)** com 35 pares no formato `.jsonl`, cada um contendo estritamente:

| Chave | Descrição |
|-------|-----------|
| `prompt` | Solicitação ou pergunta realista do usuário |
| `chosen` | Resposta segura e alinhada (HHH) |
| `rejected` | Resposta prejudicial ou inadequada |

Cobertura: injeção SQL, malware, phishing, engenharia social, LGPD/GDPR, tom corporativo inadequado, entre outros.

### Passo 2 — Pipeline DPO

O `DPOTrainer` da biblioteca `trl` (Hugging Face) requer dois modelos simultâneos:

- **Modelo Ator** (`model`): carregado com adaptadores LoRA, tem pesos **atualizáveis** — é ele que aprende as preferências humanas.
- **Modelo de Referência** (`ref_model`): carregado sem LoRA, com todos os parâmetros **congelados** — serve exclusivamente para calcular a divergência Kullback-Leibler (KL) e garantir que o ator não se afaste demais da distribuição original.

### Passo 3 — Hiperparâmetro Beta (β)

#### Justificativa Matemática do β no DPO

Na formulação matemática do DPO (Rafailov et al., 2023), o objetivo de otimização é:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x)}{\pi_{ref}(y_w | x)} - \beta \log \frac{\pi_\theta(y_l | x)}{\pi_{ref}(y_l | x)} \right) \right]$$

O parâmetro **β** funciona matematicamente como um **coeficiente de penalidade KL** — um "imposto" proporcional à distância entre o modelo ator (π_θ) e o modelo de referência (π_ref). Quanto maior o β, mais caro fica para o modelo ator se desviar da distribuição original: o gradiente do termo KL cresce, forçando o ator a permanecer próximo ao comportamento de base. Com **β = 0.1**, escolhemos uma penalidade leve que permite ao modelo aprender com força suficiente a suprimir respostas tóxicas (aumentando log-prob de `chosen` e diminuindo a de `rejected`), mas sem sacrificar a fluência e a coerência linguística adquiridas durante o pré-treinamento. Em outras palavras, β age como um regulador: valores muito baixos (β → 0) ignoram a referência e podem colapsar a fluência do modelo; valores muito altos (β → ∞) travam o aprendizado de preferências. β = 0.1 é o ponto de equilíbrio amplamente utilizado na literatura para tarefas de alinhamento de segurança em modelos de pequeno a médio porte.

### Passo 4 — Treinamento e Validação

- Otimizador: `paged_adamw_32bit` (economia de VRAM)
- Precisão mista: `fp16=True`
- Gradient checkpointing habilitado
- Validação: comparação de log-probabilidades entre `chosen` e `rejected` para um prompt malicioso pós-treinamento

---

## Nota de IA

> Partes geradas/complementadas com IA, revisadas por Ingrid.
