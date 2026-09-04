# physics-thesis-2d-turbulence
# Pseudospectral Simulations of 2D Navier-Stokes Turbulence

Repository contenente il codice numerico, il setup OpenFOAM e gli script di post-processing sviluppati per la tesi di laurea triennale in Fisica:

> **Titolo Tesi:** Turbolenza di due dimensioni  
> **Candidato:** Andrea Paccone  
> **Relatore:** Prof. Gianmaria Falasco  
> **Istituzione:** Università degli Studi di Padova – Dipartimento di Fisica e Astronomia "G. Galilei"  
> **Anno Accademico:** 2025/2026  

---

## Panoramica del Progetto

Il progetto implementa e confronta simulazioni numeriche della turbolenza bidimensionale retta dalle equazioni di Navier-Stokes in regime incomprimibile. Il repository include:
- Un solver bidimensionale basato su **metodi pseudospettrali** (FFTW3, integrazione temporale Runge-Kutta, dealiasing 2/3).
- Il caso di configurazione per simulazioni in strato sottile (*thin-layer*) con **OpenFOAM** (`pisoFoam`).
- Script Python e Jupyter Notebook per il calcolo di spettri di energia, enstrofia e analisi statistica dei campi di vorticità.

---

## Struttura della Repository

```

├── solver/         # Codice sorgente del solver pseudospettrale (C++)
├── openfoam/       # Setup pulito del caso OpenFOAM (0, constant, system)
├── analysis/       # Script Python per calcolo spettri e grafici
├── notebooks/      # Jupyter Notebook per l'analisi dati interattiva
├── data/           # Directory di destinazione per gli output di simulazione
└── README.md
