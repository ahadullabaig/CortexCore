# CortexCore Frontend Redesign
## Dark Neuroscience Aesthetic | Minimal & Distinctive

**Version:** 2.0
**Target Audience:** Research/Technical Users
**Core Philosophy:** Brain-inspired cyberpunk interface with clinical precision
**Anti-Pattern:** Avoid generic "AI slop" aesthetics (purple gradients, system fonts, cookie-cutter layouts)

---

## Table of Contents

1. [Current State Analysis](#current-state-analysis)
2. [Design Vision](#design-vision)
3. [Typography System](#typography-system)
4. [Color Architecture](#color-architecture)
5. [Animation & Motion](#animation--motion)
6. [Background & Depth](#background--depth)
7. [Component Redesign](#component-redesign)
8. [Implementation Phases](#implementation-phases)
9. [Technical Specifications](#technical-specifications)
10. [Performance Considerations](#performance-considerations)

---

## Current State Analysis

### Identified "AI Slop" Issues

**❌ Generic Purple Gradient**
```css
/* Current */
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
```
- Overused in AI-generated designs
- Lacks contextual meaning
- No connection to neuroscience/medical domain

**❌ System Font Stack**
```css
/* Current */
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto'...;
```
- Safe but uninspired
- No distinctive character
- Doesn't convey technical precision

**❌ Minimal Animation**
- Only probability bar width transitions (0.5s)
- No page load choreography
- No spike firing animations
- Generic card hover effects

**❌ Flat Backgrounds**
- Solid colors and simple gradients
- No atmospheric depth
- Missing neural network patterns
- No contextual visual identity

### Existing Strengths to Preserve

✅ **Clean Component Hierarchy**
- Logical flow: Generation → Visualization → Prediction → Results
- Card-based layout works well for technical users
- Responsive grid system

✅ **Plotly Integration**
- Professional scientific visualizations
- Interactive plots for data exploration
- Good performance with large datasets

✅ **Real Model Integration**
- Actually uses trained SNNs (not just mockups)
- Live inference with timing metrics
- Proper error handling

---

## Design Vision

### Core Concept: "Neural Observatory"

The interface should feel like a **high-tech neural activity monitoring station** — part oscilloscope, part brain scanner, part cyberpunk research terminal. Think: clinical precision meets neuromorphic computing meets dark mode excellence.

### Aesthetic Pillars

1. **Dark Neuroscience** — Deep backgrounds with electric neural accents
2. **Minimal Brutalism** — Essential elements only, no decoration for decoration's sake
3. **Data Primacy** — Visualizations are the hero, UI recedes into darkness
4. **Dynamic Life** — Neural activity feels alive through animation
5. **Clinical Precision** — Medical-grade accuracy conveyed through typography and layout

### Mood Board References

- **Neuroscience Lab Equipment** — Oscilloscopes, EEG monitors, patch-clamp rigs
- **IDE Dark Themes** — VSCode "One Dark Pro", Sublime "Monokai Pro"
- **Cyberpunk Interfaces** — Ghost in the Shell, Blade Runner 2049 UI
- **Medical Imaging** — MRI/CT scan displays, patient monitors
- **Neural Simulation Software** — NEURON, Brian2, NetPyNE interfaces

---

## Typography System

### Primary Font: **JetBrains Mono** (Monospace)

**Why:** Technical precision, excellent readability, distinctive character, open-source
**Usage:** Headers, labels, metrics, code-like elements
**Weights:** 400 (Regular), 600 (SemiBold), 700 (Bold)

```css
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');
```

### Secondary Font: **Archivo** (Sans-serif)

**Why:** Clinical clarity, modern, geometric precision, excellent at small sizes
**Usage:** Body text, descriptions, secondary information
**Weights:** 400 (Regular), 500 (Medium)

```css
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400;500&display=swap');
```

### Accent Font: **Rajdhani** (Display Sans)

**Why:** Futuristic, technical, high-tech feel, excellent for headings
**Usage:** Page title, large numbers, hero metrics
**Weights:** 600 (SemiBold), 700 (Bold)

```css
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&display=swap');
```

### Typography Scale

```css
:root {
  /* Font Families */
  --font-display: 'Rajdhani', sans-serif;
  --font-primary: 'JetBrains Mono', monospace;
  --font-secondary: 'Archivo', sans-serif;

  /* Type Scale */
  --text-xs: 0.75rem;    /* 12px - Labels, captions */
  --text-sm: 0.875rem;   /* 14px - Body small */
  --text-base: 1rem;     /* 16px - Body */
  --text-lg: 1.125rem;   /* 18px - Subheadings */
  --text-xl: 1.5rem;     /* 24px - Card headers */
  --text-2xl: 2rem;      /* 32px - Metrics */
  --text-3xl: 2.5rem;    /* 40px - Hero title */
  --text-4xl: 3.5rem;    /* 56px - Large numbers */
}
```

### Anti-Pattern Avoidance

❌ **DO NOT USE:**
- Inter, Roboto, Arial, Helvetica Neue
- System UI fonts (too generic)
- Space Grotesk, Poppins (overused in AI designs)

---

## Color Architecture

### Dark Neuroscience Palette

Inspired by: Neural network visualizations, oscilloscope displays, clinical monitors

```css
:root {
  /* Base Colors - Deep Space Background */
  --bg-primary: #0a0e1a;      /* Almost black, slight blue tint */
  --bg-secondary: #0f1419;    /* Cards, elevated surfaces */
  --bg-tertiary: #1a1f2e;     /* Hover states, inputs */

  /* Neural Electric Accents */
  --neural-cyan: #00d9ff;     /* Primary accent - synaptic activity */
  --neural-blue: #0088ff;     /* Secondary accent - action potentials */
  --neural-purple: #a855f7;   /* Tertiary - deep brain activity */
  --neural-teal: #14b8a6;     /* Quaternary - stable states */

  /* Clinical Status Colors */
  --clinical-normal: #10b981; /* Green - healthy signals */
  --clinical-warning: #f59e0b;/* Amber - attention required */
  --clinical-critical: #ef4444;/* Red - arrhythmia/abnormal */
  --clinical-info: #3b82f6;   /* Blue - informational */

  /* Text Colors */
  --text-primary: #e4e7ed;    /* High contrast white-ish */
  --text-secondary: #9ca3af;  /* Dimmed for secondary info */
  --text-tertiary: #6b7280;   /* Even more subtle */
  --text-accent: #00d9ff;     /* Cyan for emphasis */

  /* Border & Divider */
  --border-subtle: #1f2937;   /* Barely visible */
  --border-moderate: #374151; /* Visible but not dominant */
  --border-strong: #4b5563;   /* Clear separation */
  --border-glow: rgba(0, 217, 255, 0.3); /* Cyan glow effect */

  /* Glow Effects */
  --glow-cyan: 0 0 20px rgba(0, 217, 255, 0.5);
  --glow-blue: 0 0 20px rgba(0, 136, 255, 0.4);
  --glow-purple: 0 0 20px rgba(168, 85, 247, 0.4);
  --glow-green: 0 0 20px rgba(16, 185, 129, 0.4);
  --glow-red: 0 0 20px rgba(239, 68, 68, 0.4);
}
```

### Gradient Backgrounds (Neural Network Patterns)

```css
:root {
  /* Subtle Background Gradients */
  --gradient-bg: radial-gradient(
    ellipse at top,
    rgba(0, 217, 255, 0.05) 0%,
    transparent 50%
  );

  /* Card Backgrounds with Neural Depth */
  --gradient-card: linear-gradient(
    135deg,
    rgba(15, 20, 25, 0.95) 0%,
    rgba(10, 14, 26, 0.98) 100%
  );

  /* Accent Gradients */
  --gradient-accent: linear-gradient(
    90deg,
    var(--neural-cyan) 0%,
    var(--neural-blue) 100%
  );

  --gradient-clinical: linear-gradient(
    90deg,
    var(--clinical-normal) 0%,
    var(--neural-teal) 100%
  );
}
```

### Color Usage Guidelines

**Primary Accent (Cyan `#00d9ff`):**
- Spike visualization markers
- Active state indicators
- Interactive element highlights
- Primary CTA buttons

**Secondary Accent (Blue `#0088ff`):**
- Links and secondary CTAs
- Hover states
- Selection indicators

**Clinical Colors:**
- Normal ECG: Green (`#10b981`)
- Arrhythmia ECG: Red (`#ef4444`)
- Status badges: Contextual (green/amber/red)

**Neural Purple:**
- Advanced features
- Experimental controls
- Neural network depth indicators

---

## Animation & Motion

### Animation Philosophy: "Neural Pulse"

Animations should feel **organic, electrical, and alive** — like watching real neural activity under a microscope. Every motion should serve a purpose: guiding attention, indicating state changes, or conveying data relationships.

### Timing System

```css
:root {
  /* Duration */
  --duration-instant: 100ms;  /* Immediate feedback */
  --duration-fast: 200ms;     /* Micro-interactions */
  --duration-base: 300ms;     /* Standard transitions */
  --duration-slow: 500ms;     /* Emphasis */
  --duration-slower: 800ms;   /* Dramatic reveals */

  /* Easing Functions */
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-out: cubic-bezier(0, 0, 0.2, 1);
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  --ease-elastic: cubic-bezier(0.68, -0.55, 0.265, 1.55);
}
```

### Page Load Choreography

**Staggered Card Reveals** (High-Impact Moment)

```css
@keyframes slideUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.card {
  animation: slideUp var(--duration-slow) var(--ease-out);
}

.card:nth-child(1) { animation-delay: 100ms; }
.card:nth-child(2) { animation-delay: 200ms; }
.card:nth-child(3) { animation-delay: 300ms; }
.card:nth-child(4) { animation-delay: 400ms; }
.card:nth-child(5) { animation-delay: 500ms; }
```

### Spike Animation System

**Real-Time Spike Firing Effect**

```css
@keyframes spikeFire {
  0% {
    opacity: 0;
    transform: scale(0.5);
  }
  50% {
    opacity: 1;
    transform: scale(1.2);
    filter: drop-shadow(var(--glow-cyan));
  }
  100% {
    opacity: 0.8;
    transform: scale(1);
  }
}

.spike-marker {
  animation: spikeFire 300ms var(--ease-spring);
}
```

**Spike Trail Effect** (After spike fires)

```css
@keyframes spikeTrail {
  from {
    opacity: 1;
    box-shadow: var(--glow-cyan);
  }
  to {
    opacity: 0;
    box-shadow: 0 0 0 rgba(0, 217, 255, 0);
  }
}

.spike-marker::after {
  content: '';
  animation: spikeTrail 1s ease-out forwards;
}
```

### ECG Trace Animation

**Drawing Effect** (When signal is generated)

```css
@keyframes drawTrace {
  from {
    stroke-dashoffset: 2500;
  }
  to {
    stroke-dashoffset: 0;
  }
}

.ecg-line {
  stroke-dasharray: 2500;
  animation: drawTrace 2s ease-out forwards;
}
```

### Button Micro-Interactions

```css
.btn {
  position: relative;
  overflow: hidden;
  transition: all var(--duration-base) var(--ease-in-out);
}

.btn::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 50%;
  width: 0;
  height: 0;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  transition: width var(--duration-slow), height var(--duration-slow);
}

.btn:hover::before {
  width: 300px;
  height: 300px;
}

.btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 20px rgba(0, 217, 255, 0.3);
}

.btn:active {
  transform: translateY(0);
}
```

### Status Badge Pulse

```css
@keyframes statusPulse {
  0%, 100% {
    box-shadow: 0 0 0 0 currentColor;
    opacity: 1;
  }
  50% {
    box-shadow: 0 0 10px 5px currentColor;
    opacity: 0.8;
  }
}

.status-badge.success {
  animation: statusPulse 2s ease-in-out infinite;
}
```

### Neural Network Background Animation

```css
@keyframes neuralPulse {
  0%, 100% {
    opacity: 0.3;
  }
  50% {
    opacity: 0.6;
  }
}

.neural-bg-layer {
  animation: neuralPulse 8s ease-in-out infinite;
}
```

### Loading States

**Neural Activity Loader**

```css
@keyframes neuronFire {
  0%, 100% {
    background: var(--neural-cyan);
    transform: scale(1);
  }
  50% {
    background: var(--neural-purple);
    transform: scale(1.5);
    box-shadow: var(--glow-cyan);
  }
}

.loading-neuron {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin: 0 4px;
  animation: neuronFire 1.4s ease-in-out infinite;
}

.loading-neuron:nth-child(1) { animation-delay: 0s; }
.loading-neuron:nth-child(2) { animation-delay: 0.2s; }
.loading-neuron:nth-child(3) { animation-delay: 0.4s; }
```

---

## Background & Depth

### Layered Background System

**Layer 1: Base Color**
```css
body {
  background-color: var(--bg-primary);
}
```

**Layer 2: Subtle Gradient Overlay**
```css
body::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: radial-gradient(
    ellipse at top right,
    rgba(0, 217, 255, 0.05) 0%,
    transparent 50%
  );
  pointer-events: none;
  z-index: -1;
}
```

**Layer 3: Neural Network Pattern**

SVG pattern for subtle neural connectivity visualization:

```svg
<svg width="100%" height="100%" style="position: fixed; top: 0; left: 0; opacity: 0.03; z-index: -2;">
  <defs>
    <pattern id="neuralNetwork" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
      <!-- Nodes -->
      <circle cx="10" cy="10" r="2" fill="#00d9ff" />
      <circle cx="90" cy="30" r="2" fill="#00d9ff" />
      <circle cx="50" cy="70" r="2" fill="#00d9ff" />
      <circle cx="30" cy="90" r="2" fill="#00d9ff" />

      <!-- Connections -->
      <line x1="10" y1="10" x2="90" y2="30" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="90" y1="30" x2="50" y2="70" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="50" y1="70" x2="30" y2="90" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="10" y1="10" x2="30" y2="90" stroke="#00d9ff" stroke-width="0.5" opacity="0.2" />
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#neuralNetwork)" />
</svg>
```

**Layer 4: Scan Line Effect** (Optional, subtle)

```css
body::after {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0, 217, 255, 0.01) 2px,
    rgba(0, 217, 255, 0.01) 4px
  );
  pointer-events: none;
  z-index: 100;
  animation: scanLine 10s linear infinite;
}

@keyframes scanLine {
  from { transform: translateY(0); }
  to { transform: translateY(100vh); }
}
```

### Card Depth System

```css
.card {
  background: var(--gradient-card);
  backdrop-filter: blur(10px);
  border: 1px solid var(--border-moderate);
  box-shadow:
    0 4px 6px rgba(0, 0, 0, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
  position: relative;
}

.card::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: linear-gradient(
    135deg,
    rgba(0, 217, 255, 0.2),
    transparent 50%,
    rgba(168, 85, 247, 0.1)
  );
  -webkit-mask:
    linear-gradient(#fff 0 0) content-box,
    linear-gradient(#fff 0 0);
  mask:
    linear-gradient(#fff 0 0) content-box,
    linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
}

.card:hover {
  border-color: var(--border-glow);
  box-shadow:
    0 8px 16px rgba(0, 0, 0, 0.5),
    0 0 20px rgba(0, 217, 255, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}
```

---

## Component Redesign

### 1. Header: "Neural Observatory Control"

**Before:** Generic purple gradient header with emoji brain
**After:** Dark oscilloscope-style display with animated neural activity indicator

```html
<header class="neural-header">
  <div class="container">
    <div class="neural-title-group">
      <div class="neural-pulse-icon">
        <span class="pulse-dot"></span>
        <span class="pulse-dot"></span>
        <span class="pulse-dot"></span>
      </div>
      <h1 class="neural-title">CORTEXCORE</h1>
      <div class="neural-version">v2.0</div>
    </div>
    <p class="neural-subtitle">
      <span class="subtitle-label">SYSTEM:</span>
      Neuromorphic Signal Processing & Classification
    </p>
  </div>
</header>
```

```css
.neural-header {
  background: var(--bg-secondary);
  border-bottom: 2px solid var(--border-moderate);
  padding: 2rem 0;
  position: relative;
  overflow: hidden;
}

.neural-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: var(--gradient-accent);
  box-shadow: var(--glow-cyan);
}

.neural-title-group {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
}

.neural-pulse-icon {
  display: flex;
  gap: 4px;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--neural-cyan);
  animation: neuronFire 1.4s ease-in-out infinite;
}

.neural-title {
  font-family: var(--font-display);
  font-size: var(--text-3xl);
  font-weight: 700;
  letter-spacing: 0.1em;
  color: var(--text-primary);
  text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}

.neural-version {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  padding: 0.25rem 0.5rem;
  border: 1px solid var(--border-subtle);
  border-radius: 4px;
}

.neural-subtitle {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.subtitle-label {
  color: var(--neural-cyan);
  font-weight: 600;
}
```

### 2. System Status: "Mission Control Panel"

**Before:** Generic colored badges
**After:** Oscilloscope-style status indicators with live metrics

```html
<div class="card status-panel">
  <h2 class="card-title">
    <span class="title-icon">⚡</span>
    SYSTEM STATUS
  </h2>
  <div class="status-grid-enhanced">
    <!-- Server Status -->
    <div class="status-metric">
      <div class="metric-label">
        <span class="label-text">SERVER</span>
        <span class="label-indicator online"></span>
      </div>
      <div class="metric-value">ONLINE</div>
      <div class="metric-graph">
        <div class="graph-bar" style="height: 95%"></div>
      </div>
    </div>

    <!-- Model Status -->
    <div class="status-metric">
      <div class="metric-label">
        <span class="label-text">MODEL</span>
        <span id="model-indicator" class="label-indicator loading"></span>
      </div>
      <div id="model-status-value" class="metric-value">LOADING...</div>
      <div class="metric-detail" id="model-arch">SimpleSNN</div>
    </div>

    <!-- Device Status -->
    <div class="status-metric">
      <div class="metric-label">
        <span class="label-text">DEVICE</span>
        <span id="device-indicator" class="label-indicator"></span>
      </div>
      <div id="device-status-value" class="metric-value">-</div>
      <div class="metric-detail" id="device-memory">0 GB VRAM</div>
    </div>
  </div>
</div>
```

```css
.status-panel {
  background: var(--gradient-card);
}

.card-title {
  font-family: var(--font-primary);
  font-size: var(--text-lg);
  font-weight: 600;
  color: var(--text-accent);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-subtle);
}

.title-icon {
  font-size: 1.2em;
  filter: drop-shadow(var(--glow-cyan));
}

.status-grid-enhanced {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
}

.status-metric {
  background: rgba(26, 31, 46, 0.5);
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 1rem;
  transition: all var(--duration-base);
}

.status-metric:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 10px rgba(0, 217, 255, 0.1);
}

.metric-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.5rem;
}

.label-text {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  font-weight: 600;
  letter-spacing: 0.1em;
}

.label-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  position: relative;
}

.label-indicator.online {
  background: var(--clinical-normal);
  box-shadow: var(--glow-green);
  animation: statusPulse 2s ease-in-out infinite;
}

.label-indicator.loading {
  background: var(--clinical-warning);
  animation: statusPulse 1s ease-in-out infinite;
}

.metric-value {
  font-family: var(--font-primary);
  font-size: var(--text-xl);
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 0.25rem;
}

.metric-detail {
  font-family: var(--font-secondary);
  font-size: var(--text-xs);
  color: var(--text-secondary);
}

.metric-graph {
  height: 4px;
  background: var(--border-moderate);
  border-radius: 2px;
  margin-top: 0.5rem;
  overflow: hidden;
}

.graph-bar {
  height: 100%;
  background: var(--gradient-accent);
  border-radius: 2px;
  transition: width var(--duration-slow);
}
```

### 3. Signal Generation Controls: "Neural Encoder Panel"

**Before:** Standard dropdown + buttons
**After:** Tactical control interface with parameter preview

```html
<div class="card encoder-panel">
  <h2 class="card-title">
    <span class="title-icon">🎛️</span>
    SIGNAL GENERATOR
  </h2>

  <div class="encoder-controls">
    <div class="control-section">
      <label class="control-label">
        <span class="label-text">CONDITION</span>
        <span class="label-hint">Select target pattern</span>
      </label>
      <div class="control-select-wrapper">
        <select id="condition-select" class="control-select">
          <option value="normal">Normal Sinus Rhythm</option>
          <option value="arrhythmia">Cardiac Arrhythmia</option>
        </select>
        <div class="select-arrow">▼</div>
      </div>
      <div class="control-preview">
        <span class="preview-label">Expected:</span>
        <span id="condition-preview" class="preview-value">70 BPM, Low Noise</span>
      </div>
    </div>

    <div class="control-actions">
      <button id="generate-btn" class="btn btn-neural">
        <span class="btn-icon">⚡</span>
        <span class="btn-text">GENERATE SIGNAL</span>
      </button>
      <button id="predict-btn" class="btn btn-clinical" disabled>
        <span class="btn-icon">🔬</span>
        <span class="btn-text">RUN INFERENCE</span>
      </button>
    </div>
  </div>
</div>
```

```css
.encoder-panel {
  background: var(--gradient-card);
}

.encoder-controls {
  display: grid;
  gap: 1.5rem;
}

.control-section {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.control-label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.label-text {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--text-primary);
  letter-spacing: 0.05em;
}

.label-hint {
  font-family: var(--font-secondary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
}

.control-select-wrapper {
  position: relative;
}

.control-select {
  font-family: var(--font-secondary);
  font-size: var(--text-base);
  color: var(--text-primary);
  background: var(--bg-tertiary);
  border: 2px solid var(--border-moderate);
  border-radius: 6px;
  padding: 0.75rem 3rem 0.75rem 1rem;
  width: 100%;
  cursor: pointer;
  transition: all var(--duration-base);
  appearance: none;
}

.control-select:hover {
  border-color: var(--border-strong);
}

.control-select:focus {
  outline: none;
  border-color: var(--neural-cyan);
  box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1);
}

.select-arrow {
  position: absolute;
  right: 1rem;
  top: 50%;
  transform: translateY(-50%);
  color: var(--text-tertiary);
  pointer-events: none;
  font-size: var(--text-xs);
}

.control-preview {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.5rem 1rem;
  background: rgba(0, 217, 255, 0.05);
  border: 1px solid rgba(0, 217, 255, 0.2);
  border-radius: 4px;
}

.preview-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  text-transform: uppercase;
}

.preview-value {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--neural-cyan);
  font-weight: 600;
}

.control-actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}

/* Buttons */
.btn {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  font-weight: 600;
  letter-spacing: 0.05em;
  padding: 0.875rem 1.5rem;
  border: 2px solid;
  border-radius: 6px;
  cursor: pointer;
  transition: all var(--duration-base);
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.5rem;
  position: relative;
  overflow: hidden;
}

.btn::before {
  content: '';
  position: absolute;
  inset: 0;
  background: rgba(255, 255, 255, 0.1);
  transform: translateX(-100%);
  transition: transform var(--duration-base);
}

.btn:hover::before {
  transform: translateX(0);
}

.btn-neural {
  background: transparent;
  border-color: var(--neural-cyan);
  color: var(--neural-cyan);
}

.btn-neural:hover:not(:disabled) {
  background: var(--neural-cyan);
  color: var(--bg-primary);
  box-shadow: var(--glow-cyan);
  transform: translateY(-2px);
}

.btn-clinical {
  background: transparent;
  border-color: var(--clinical-normal);
  color: var(--clinical-normal);
}

.btn-clinical:hover:not(:disabled) {
  background: var(--clinical-normal);
  color: var(--bg-primary);
  box-shadow: var(--glow-green);
  transform: translateY(-2px);
}

.btn:disabled {
  opacity: 0.3;
  cursor: not-allowed;
  border-color: var(--border-moderate);
  color: var(--text-tertiary);
}

.btn:active:not(:disabled) {
  transform: translateY(0);
}

.btn-icon {
  font-size: 1.2em;
  filter: drop-shadow(0 0 4px currentColor);
}
```

### 4. ECG Visualization: "Neural Signal Oscilloscope"

**Before:** Plotly chart with default styling
**After:** Dark oscilloscope with scan line effects and metadata overlay

```html
<div class="card oscilloscope-card">
  <div class="oscilloscope-header">
    <h2 class="card-title">
      <span class="title-icon">📡</span>
      ECG SIGNAL
    </h2>
    <div class="oscilloscope-metadata">
      <span class="metadata-item">
        <span class="metadata-label">SR:</span>
        <span class="metadata-value">250 Hz</span>
      </span>
      <span class="metadata-item">
        <span class="metadata-label">DUR:</span>
        <span class="metadata-value">10.0 s</span>
      </span>
      <span class="metadata-item">
        <span class="metadata-label">PTS:</span>
        <span id="ecg-points" class="metadata-value">2500</span>
      </span>
    </div>
  </div>
  <div class="oscilloscope-wrapper">
    <div id="ecg-plot" class="plot-oscilloscope"></div>
  </div>
</div>
```

```css
.oscilloscope-card {
  background: var(--gradient-card);
}

.oscilloscope-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
  margin-bottom: 1rem;
}

.oscilloscope-metadata {
  display: flex;
  gap: 1.5rem;
}

.metadata-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.metadata-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  font-weight: 600;
}

.metadata-value {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--neural-cyan);
  font-weight: 700;
}

.oscilloscope-wrapper {
  position: relative;
  background: rgba(0, 0, 0, 0.5);
  border: 2px solid var(--border-moderate);
  border-radius: 8px;
  padding: 0.5rem;
  overflow: hidden;
}

.oscilloscope-wrapper::before {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 20px,
    rgba(0, 217, 255, 0.03) 20px,
    rgba(0, 217, 255, 0.03) 21px
  );
  pointer-events: none;
  z-index: 1;
}

.plot-oscilloscope {
  width: 100%;
  height: 400px;
  position: relative;
  z-index: 2;
}
```

**Plotly Styling Override:**

```javascript
const ecgLayout = {
  paper_bgcolor: 'rgba(0, 0, 0, 0)',
  plot_bgcolor: 'rgba(0, 0, 0, 0)',
  font: {
    family: 'JetBrains Mono, monospace',
    color: '#9ca3af',
    size: 12
  },
  xaxis: {
    title: { text: 'Sample', font: { color: '#6b7280' } },
    gridcolor: '#1f2937',
    zerolinecolor: '#374151'
  },
  yaxis: {
    title: { text: 'Amplitude', font: { color: '#6b7280' } },
    gridcolor: '#1f2937',
    zerolinecolor: '#374151'
  },
  margin: { t: 20, r: 20, b: 40, l: 50 }
};

const ecgTrace = {
  y: signalData,
  type: 'scatter',
  mode: 'lines',
  line: {
    color: condition === 'normal' ? '#10b981' : '#ef4444',
    width: 2,
    shape: 'spline'
  },
  hovertemplate: '<b>Sample</b>: %{x}<br><b>Amplitude</b>: %{y:.4f}<extra></extra>'
};
```

### 5. Spike Raster: "Neural Activity Monitor"

**Hero Element** — This is the centerpiece of the interface

```html
<div class="card spike-monitor-card">
  <div class="spike-monitor-header">
    <h2 class="card-title">
      <span class="title-icon">⚡</span>
      SPIKE RASTER
      <span class="title-badge">NEUROMORPHIC ENCODING</span>
    </h2>
    <div class="spike-stats">
      <div class="stat-item">
        <span class="stat-label">TOTAL SPIKES:</span>
        <span id="total-spikes" class="stat-value">-</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">FIRING RATE:</span>
        <span id="firing-rate" class="stat-value">-</span>
      </div>
      <div class="stat-item">
        <span class="stat-label">SPARSITY:</span>
        <span id="sparsity" class="stat-value">-</span>
      </div>
    </div>
  </div>
  <div class="spike-monitor-wrapper">
    <div id="spike-plot" class="plot-spike-raster"></div>
    <div class="spike-overlay">
      <div class="spike-legend">
        <div class="legend-item">
          <div class="legend-marker active"></div>
          <span class="legend-text">Active Neuron</span>
        </div>
        <div class="legend-item">
          <div class="legend-marker silent"></div>
          <span class="legend-text">Silent Neuron</span>
        </div>
      </div>
    </div>
  </div>
</div>
```

```css
.spike-monitor-card {
  background: var(--gradient-card);
  border: 2px solid var(--border-moderate);
}

.spike-monitor-card:hover {
  border-color: var(--border-glow);
  box-shadow:
    0 8px 20px rgba(0, 0, 0, 0.5),
    0 0 30px rgba(0, 217, 255, 0.15);
}

.spike-monitor-header {
  display: flex;
  flex-direction: column;
  gap: 1rem;
  margin-bottom: 1.5rem;
}

.title-badge {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  font-weight: 600;
  color: var(--neural-purple);
  background: rgba(168, 85, 247, 0.1);
  border: 1px solid rgba(168, 85, 247, 0.3);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  margin-left: auto;
}

.spike-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  padding: 1rem;
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border-subtle);
  border-radius: 6px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.stat-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  font-weight: 600;
}

.stat-value {
  font-family: var(--font-primary);
  font-size: var(--text-xl);
  color: var(--neural-cyan);
  font-weight: 700;
  text-shadow: 0 0 10px rgba(0, 217, 255, 0.5);
}

.spike-monitor-wrapper {
  position: relative;
  background: rgba(0, 0, 0, 0.7);
  border: 2px solid var(--neural-cyan);
  border-radius: 8px;
  padding: 1rem;
  box-shadow:
    inset 0 0 20px rgba(0, 217, 255, 0.1),
    var(--glow-cyan);
}

.spike-monitor-wrapper::before {
  content: '';
  position: absolute;
  inset: 0;
  background:
    repeating-linear-gradient(
      0deg,
      transparent,
      transparent 10px,
      rgba(0, 217, 255, 0.02) 10px,
      rgba(0, 217, 255, 0.02) 11px
    ),
    repeating-linear-gradient(
      90deg,
      transparent,
      transparent 10px,
      rgba(0, 217, 255, 0.02) 10px,
      rgba(0, 217, 255, 0.02) 11px
    );
  pointer-events: none;
  z-index: 1;
}

.plot-spike-raster {
  width: 100%;
  height: 500px;
  position: relative;
  z-index: 2;
}

.spike-overlay {
  position: absolute;
  top: 1rem;
  right: 1rem;
  z-index: 10;
}

.spike-legend {
  background: rgba(10, 14, 26, 0.9);
  border: 1px solid var(--border-moderate);
  border-radius: 6px;
  padding: 0.75rem;
  backdrop-filter: blur(10px);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-bottom: 0.5rem;
}

.legend-item:last-child {
  margin-bottom: 0;
}

.legend-marker {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-marker.active {
  background: var(--neural-cyan);
  box-shadow: var(--glow-cyan);
  animation: statusPulse 2s ease-in-out infinite;
}

.legend-marker.silent {
  background: var(--border-moderate);
}

.legend-text {
  font-family: var(--font-secondary);
  font-size: var(--text-xs);
  color: var(--text-secondary);
}
```

**Plotly Spike Raster Styling:**

```javascript
const spikeTrace = {
  x: spikeData.spike_times,
  y: spikeData.neuron_ids,
  mode: 'markers',
  type: 'scatter',
  marker: {
    color: '#00d9ff',
    size: 6,
    symbol: 'line-ns-open',
    line: { width: 2 },
    opacity: 0.8
  },
  hovertemplate: '<b>Time</b>: %{x}<br><b>Neuron</b>: %{y}<extra></extra>'
};

const spikeLayout = {
  paper_bgcolor: 'rgba(0, 0, 0, 0)',
  plot_bgcolor: 'rgba(0, 0, 0, 0)',
  font: {
    family: 'JetBrains Mono, monospace',
    color: '#9ca3af',
    size: 12
  },
  xaxis: {
    title: { text: 'Time Step', font: { color: '#6b7280' } },
    gridcolor: '#1f2937',
    zerolinecolor: '#374151',
    range: [0, spikeData.num_steps]
  },
  yaxis: {
    title: { text: 'Neuron Index', font: { color: '#6b7280' } },
    gridcolor: '#1f2937',
    zerolinecolor: '#374151',
    range: [-1, spikeData.num_neurons]
  },
  margin: { t: 20, r: 20, b: 40, l: 50 }
};
```

### 6. Prediction Results: "Clinical Analysis Dashboard"

**Before:** Simple result display
**After:** Diagnostic readout with confidence visualization and clinical context

```html
<div class="card results-dashboard" id="results-card" style="display: none;">
  <div class="results-header">
    <h2 class="card-title">
      <span class="title-icon">🔬</span>
      INFERENCE RESULTS
    </h2>
    <div class="results-timestamp">
      <span class="timestamp-label">COMPUTED:</span>
      <span id="inference-timestamp" class="timestamp-value">-</span>
    </div>
  </div>

  <div class="results-primary">
    <div class="classification-panel">
      <div class="classification-label">CLASSIFICATION</div>
      <div id="prediction-class" class="classification-value">-</div>
      <div class="classification-confidence">
        <span class="confidence-label">CONFIDENCE:</span>
        <span id="prediction-confidence" class="confidence-value">-</span>
      </div>
    </div>

    <div class="metrics-panel">
      <div class="metric-box">
        <div class="metric-box-label">INFERENCE TIME</div>
        <div id="inference-time" class="metric-box-value">-</div>
      </div>
      <div class="metric-box">
        <div class="metric-box-label">SPIKES PROCESSED</div>
        <div id="spikes-processed" class="metric-box-value">-</div>
      </div>
    </div>
  </div>

  <div class="probability-section">
    <h3 class="section-subtitle">CLASS PROBABILITIES</h3>
    <div class="prob-bar-group">
      <div class="prob-bar-item">
        <div class="prob-bar-header">
          <span class="prob-bar-label">NORMAL</span>
          <span id="prob-normal-text" class="prob-bar-percentage">0%</span>
        </div>
        <div class="prob-bar-track">
          <div id="prob-normal" class="prob-bar-fill normal" style="width: 0%">
            <div class="prob-bar-glow"></div>
          </div>
        </div>
      </div>

      <div class="prob-bar-item">
        <div class="prob-bar-header">
          <span class="prob-bar-label">ARRHYTHMIA</span>
          <span id="prob-arrhythmia-text" class="prob-bar-percentage">0%</span>
        </div>
        <div class="prob-bar-track">
          <div id="prob-arrhythmia" class="prob-bar-fill arrhythmia" style="width: 0%">
            <div class="prob-bar-glow"></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="clinical-note">
    <span class="note-icon">ℹ️</span>
    <span class="note-text">Results are for research purposes only. Not for clinical diagnosis.</span>
  </div>
</div>
```

```css
.results-dashboard {
  background: var(--gradient-card);
  border: 2px solid var(--clinical-info);
  animation: slideUp var(--duration-slow) var(--ease-out);
}

.results-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1.5rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid var(--border-subtle);
}

.results-timestamp {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.timestamp-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
}

.timestamp-value {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--text-secondary);
}

.results-primary {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 1.5rem;
  margin-bottom: 2rem;
}

.classification-panel {
  background: rgba(0, 0, 0, 0.3);
  border: 2px solid var(--border-moderate);
  border-radius: 8px;
  padding: 1.5rem;
  text-align: center;
}

.classification-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  font-weight: 600;
  letter-spacing: 0.1em;
  margin-bottom: 1rem;
}

.classification-value {
  font-family: var(--font-display);
  font-size: var(--text-4xl);
  font-weight: 700;
  color: var(--text-primary);
  margin-bottom: 1rem;
  text-shadow: 0 0 20px currentColor;
  animation: slideUp var(--duration-slow) var(--ease-spring);
}

.classification-value.normal {
  color: var(--clinical-normal);
}

.classification-value.arrhythmia {
  color: var(--clinical-critical);
}

.classification-confidence {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
}

.confidence-label {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--text-secondary);
}

.confidence-value {
  font-family: var(--font-primary);
  font-size: var(--text-2xl);
  font-weight: 700;
  color: var(--neural-cyan);
}

.metrics-panel {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.metric-box {
  background: rgba(0, 217, 255, 0.05);
  border: 1px solid rgba(0, 217, 255, 0.2);
  border-radius: 6px;
  padding: 1rem;
  text-align: center;
}

.metric-box-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  margin-bottom: 0.5rem;
}

.metric-box-value {
  font-family: var(--font-primary);
  font-size: var(--text-xl);
  font-weight: 700;
  color: var(--neural-cyan);
}

.probability-section {
  margin-bottom: 1.5rem;
}

.section-subtitle {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  color: var(--text-secondary);
  font-weight: 600;
  letter-spacing: 0.05em;
  margin-bottom: 1rem;
}

.prob-bar-group {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.prob-bar-item {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.prob-bar-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.prob-bar-label {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  font-weight: 600;
  color: var(--text-primary);
}

.prob-bar-percentage {
  font-family: var(--font-primary);
  font-size: var(--text-sm);
  font-weight: 700;
  color: var(--text-accent);
}

.prob-bar-track {
  height: 32px;
  background: rgba(0, 0, 0, 0.5);
  border: 1px solid var(--border-moderate);
  border-radius: 16px;
  overflow: hidden;
  position: relative;
}

.prob-bar-fill {
  height: 100%;
  border-radius: 16px;
  transition: width var(--duration-slow) var(--ease-out);
  position: relative;
  overflow: hidden;
}

.prob-bar-fill.normal {
  background: linear-gradient(90deg, var(--clinical-normal), var(--neural-teal));
}

.prob-bar-fill.arrhythmia {
  background: linear-gradient(90deg, var(--clinical-critical), #ff6b6b);
}

.prob-bar-glow {
  position: absolute;
  inset: 0;
  background: linear-gradient(
    90deg,
    transparent 0%,
    rgba(255, 255, 255, 0.3) 50%,
    transparent 100%
  );
  animation: barShimmer 2s ease-in-out infinite;
}

@keyframes barShimmer {
  from { transform: translateX(-100%); }
  to { transform: translateX(100%); }
}

.clinical-note {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 1rem;
  background: rgba(59, 130, 246, 0.1);
  border: 1px solid rgba(59, 130, 246, 0.3);
  border-radius: 6px;
}

.note-icon {
  font-size: 1.2em;
}

.note-text {
  font-family: var(--font-secondary);
  font-size: var(--text-sm);
  color: var(--text-secondary);
}
```

### 7. Performance Metrics: "System Performance Grid"

**Before:** Generic gradient boxes
**After:** Comparison dashboard with progress indicators

```html
<div class="card performance-grid">
  <h2 class="card-title">
    <span class="title-icon">⚡</span>
    NEUROMORPHIC ADVANTAGES
  </h2>

  <div class="performance-items">
    <div class="performance-item">
      <div class="performance-header">
        <span class="performance-label">ENERGY EFFICIENCY</span>
        <span class="performance-badge">vs CNN</span>
      </div>
      <div class="performance-value">60%</div>
      <div class="performance-comparison">
        <div class="comparison-bar snn" style="width: 60%">
          <span class="comparison-label">SNN</span>
        </div>
        <div class="comparison-bar cnn" style="width: 100%">
          <span class="comparison-label">CNN</span>
        </div>
      </div>
    </div>

    <div class="performance-item">
      <div class="performance-header">
        <span class="performance-label">MODEL ACCURACY</span>
        <span id="accuracy-badge" class="performance-badge">TEST SET</span>
      </div>
      <div id="model-accuracy" class="performance-value">89.2%</div>
      <div class="performance-detail">
        Target: 92% | MVP: 85%
      </div>
    </div>

    <div class="performance-item">
      <div class="performance-header">
        <span class="performance-label">NETWORK SPARSITY</span>
        <span class="performance-badge">INACTIVE</span>
      </div>
      <div class="performance-value">70%</div>
      <div class="sparsity-indicator">
        <div class="sparsity-fill" style="width: 70%"></div>
      </div>
    </div>
  </div>
</div>
```

```css
.performance-grid {
  background: var(--gradient-card);
}

.performance-items {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
}

.performance-item {
  background: rgba(0, 0, 0, 0.3);
  border: 1px solid var(--border-moderate);
  border-radius: 8px;
  padding: 1.5rem;
  transition: all var(--duration-base);
}

.performance-item:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 15px rgba(0, 217, 255, 0.1);
  transform: translateY(-4px);
}

.performance-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.performance-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
  font-weight: 600;
  letter-spacing: 0.05em;
}

.performance-badge {
  font-family: var(--font-primary);
  font-size: 0.65rem;
  color: var(--neural-purple);
  background: rgba(168, 85, 247, 0.1);
  border: 1px solid rgba(168, 85, 247, 0.3);
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
}

.performance-value {
  font-family: var(--font-display);
  font-size: var(--text-4xl);
  font-weight: 700;
  color: var(--neural-cyan);
  margin-bottom: 1rem;
  text-shadow: 0 0 15px rgba(0, 217, 255, 0.5);
}

.performance-detail {
  font-family: var(--font-secondary);
  font-size: var(--text-xs);
  color: var(--text-secondary);
  margin-top: 0.5rem;
}

.performance-comparison {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.comparison-bar {
  height: 24px;
  border-radius: 4px;
  display: flex;
  align-items: center;
  padding: 0 0.75rem;
  transition: width var(--duration-slow);
}

.comparison-bar.snn {
  background: var(--clinical-normal);
  box-shadow: var(--glow-green);
}

.comparison-bar.cnn {
  background: var(--border-strong);
}

.comparison-label {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  font-weight: 600;
  color: white;
}

.sparsity-indicator {
  height: 8px;
  background: var(--border-moderate);
  border-radius: 4px;
  overflow: hidden;
}

.sparsity-fill {
  height: 100%;
  background: var(--gradient-accent);
  border-radius: 4px;
  transition: width var(--duration-slow);
}
```

### 8. Footer: "System Attribution"

**Before:** Generic dark footer
**After:** Minimalist credit line with version info

```html
<footer class="neural-footer">
  <div class="container">
    <div class="footer-content">
      <div class="footer-left">
        <span class="footer-text">CORTEXCORE</span>
        <span class="footer-divider">|</span>
        <span class="footer-text">Neuromorphic Signal Processing</span>
      </div>
      <div class="footer-right">
        <span class="footer-tech">PyTorch</span>
        <span class="footer-tech">snnTorch</span>
        <span class="footer-tech">Flask</span>
      </div>
    </div>
  </div>
</footer>
```

```css
.neural-footer {
  background: var(--bg-secondary);
  border-top: 1px solid var(--border-subtle);
  padding: 1.5rem 0;
  margin-top: 4rem;
}

.footer-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 1rem;
}

.footer-left,
.footer-right {
  display: flex;
  align-items: center;
  gap: 1rem;
}

.footer-text {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-tertiary);
}

.footer-divider {
  color: var(--border-moderate);
}

.footer-tech {
  font-family: var(--font-primary);
  font-size: var(--text-xs);
  color: var(--text-secondary);
  padding: 0.25rem 0.5rem;
  background: rgba(0, 217, 255, 0.05);
  border: 1px solid rgba(0, 217, 255, 0.2);
  border-radius: 4px;
}
```

---

## Implementation Phases

### Phase 1: Foundation (Day 1 — 4-6 hours)

**Goal:** Establish dark neuroscience aesthetic and typography system

**Tasks:**
1. ✅ Replace CSS color variables with dark palette
2. ✅ Import custom fonts (JetBrains Mono, Archivo, Rajdhani)
3. ✅ Update body background with layered system
4. ✅ Apply new colors to existing components
5. ✅ Test responsive breakpoints with new styles

**Files Modified:**
- `demo/static/style.css` (complete rewrite)

**Deliverables:**
- Dark theme applied globally
- Typography system active
- No broken layouts

**Validation:**
```bash
# Visual check
make demo
# Navigate to http://localhost:5000
# Verify: dark background, custom fonts, no white flashes
```

---

### Phase 2: Component Enhancement (Day 2 — 6-8 hours)

**Goal:** Redesign each component with new markup and styling

**Tasks:**
1. ✅ Update header to "Neural Observatory Control"
2. ✅ Enhance status panel to "Mission Control"
3. ✅ Redesign controls to "Neural Encoder Panel"
4. ✅ Style oscilloscope card with metadata
5. ✅ Upgrade spike monitor to hero element
6. ✅ Rebuild results dashboard
7. ✅ Enhance performance metrics grid
8. ✅ Update footer

**Files Modified:**
- `demo/templates/index.html` (restructure markup)
- `demo/static/style.css` (add component styles)

**Deliverables:**
- All 8 components redesigned
- Markup semantic and accessible
- Consistent visual language

**Validation:**
```bash
# Check each component
make demo
# Test: generate signal → view plots → run prediction → see results
# Verify: all new styles applied, no layout breaks
```

---

### Phase 3: Animation & Motion (Day 3 — 4-6 hours)

**Goal:** Bring interface to life with dynamic animations

**Tasks:**
1. ✅ Implement page load choreography (staggered card reveals)
2. ✅ Add spike firing animations to raster plot
3. ✅ Create ECG trace drawing effect
4. ✅ Add button micro-interactions
5. ✅ Implement status badge pulse
6. ✅ Add neural background animation
7. ✅ Create loading state animations
8. ✅ Add probability bar shimmer effect

**Files Modified:**
- `demo/static/style.css` (add keyframe animations)
- `demo/static/script.js` (trigger animations on events)

**Deliverables:**
- Smooth page load experience
- Interactive micro-interactions
- Live neural activity feel

**Validation:**
```bash
# Test animations
make demo
# Refresh page: cards should slide up with stagger
# Generate signal: ECG should draw in
# Run prediction: bars should animate
# Hover buttons: should see ripple + lift effects
```

---

### Phase 4: Polish & Optimization (Day 5 — 3-4 hours)

**Goal:** Fine-tune performance, accessibility, and edge cases

**Tasks:**
1. ✅ Optimize animation performance (use `transform` and `opacity` only)
2. ✅ Add prefers-reduced-motion media queries
3. ✅ Test mobile responsiveness (all breakpoints)
4. ✅ Add ARIA labels for accessibility
5. ✅ Optimize Plotly config for dark theme
6. ✅ Test with real model (CUDA + CPU modes)
7. ✅ Cross-browser testing (Chrome, Firefox, Safari)
8. ✅ Performance audit (Lighthouse)

**Files Modified:**
- `demo/static/style.css` (add media queries, optimizations)
- `demo/templates/index.html` (add ARIA attributes)
- `demo/static/script.js` (optimize Plotly configs)

**Deliverables:**
- 60+ FPS animations
- Accessible UI (WCAG 2.1 AA)
- Mobile-friendly
- Cross-browser compatible

**Validation:**
```bash
# Performance check
make demo
# Open DevTools → Performance tab → Record page load
# Verify: 60 FPS, no jank, fast load time

# Accessibility check
# Use axe DevTools or Lighthouse
# Target: 90+ accessibility score

# Mobile check
# DevTools → Device Mode → Test various screen sizes
# Verify: no horizontal scroll, readable text, touch targets 48px+
```

---

## Technical Specifications

### CSS Architecture

**File Structure:**
```
demo/static/
├── style.css              # Main stylesheet (~1500 lines)
│   ├── Variables          # CSS custom properties
│   ├── Reset & Base       # Normalization
│   ├── Typography         # Font imports & scales
│   ├── Layout             # Container, grid
│   ├── Components         # Individual component styles
│   ├── Animations         # Keyframes & transitions
│   ├── Utilities          # Helper classes
│   └── Responsive         # Media queries
```

**Naming Convention:**
- BEM-inspired: `.component-name__element--modifier`
- Semantic: `.neural-header`, `.spike-monitor-card`
- Avoid: `.box1`, `.container-blue`, `.div-wrapper`

**CSS Custom Properties Strategy:**
```css
:root {
  /* Colors: --category-descriptor */
  --bg-primary: #0a0e1a;
  --neural-cyan: #00d9ff;
  --clinical-normal: #10b981;

  /* Typography: --font-purpose, --text-size */
  --font-display: 'Rajdhani', sans-serif;
  --text-xl: 1.5rem;

  /* Timing: --duration-descriptor, --ease-type */
  --duration-base: 300ms;
  --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);

  /* Effects: --effect-descriptor */
  --glow-cyan: 0 0 20px rgba(0, 217, 255, 0.5);
}
```

### Animation Performance Guidelines

**60 FPS Rule:**
- Only animate `transform` and `opacity` (GPU-accelerated)
- Avoid animating: `width`, `height`, `top`, `left`, `margin`, `padding`

**Use `will-change` Sparingly:**
```css
.card {
  will-change: transform;  /* Only for frequently animated elements */
}
```

**Debounce Scroll/Resize Events:**
```javascript
let resizeTimeout;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => {
    // Handle resize
  }, 150);
});
```

### Responsive Breakpoints

```css
/* Mobile First Approach */

/* Base: Mobile (320px - 767px) */
/* Already optimized */

/* Tablet (768px - 1023px) */
@media (min-width: 768px) {
  .status-grid-enhanced { grid-template-columns: repeat(3, 1fr); }
  .results-primary { grid-template-columns: 2fr 1fr; }
}

/* Desktop (1024px - 1439px) */
@media (min-width: 1024px) {
  .container { max-width: 1200px; }
  .performance-items { grid-template-columns: repeat(3, 1fr); }
}

/* Large Desktop (1440px+) */
@media (min-width: 1440px) {
  .container { max-width: 1400px; }
  .plot-oscilloscope { height: 500px; }
}
```

### Accessibility Requirements

**WCAG 2.1 AA Compliance:**

1. **Color Contrast**
   - Text on bg: minimum 4.5:1 ratio
   - Large text (18pt+): minimum 3:1 ratio
   - Test tool: https://webaim.org/resources/contrastchecker/

2. **Keyboard Navigation**
   - All interactive elements must be keyboard accessible
   - Visible focus indicators (outline or box-shadow)
   - Logical tab order

3. **Screen Readers**
   - ARIA labels for icon-only buttons
   - Semantic HTML (`<header>`, `<main>`, `<section>`)
   - Alt text for images (if added)

4. **Motion Sensitivity**
```css
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
  }
}
```

### JavaScript Integration Notes

**Plotly Dark Theme Config:**
```javascript
const darkPlotlyConfig = {
  displayModeBar: true,
  modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
  displaylogo: false,
  responsive: true
};

const darkPlotlyLayout = {
  paper_bgcolor: 'rgba(0, 0, 0, 0)',
  plot_bgcolor: 'rgba(0, 0, 0, 0)',
  font: {
    family: 'JetBrains Mono, monospace',
    color: '#9ca3af',
    size: 12
  },
  xaxis: {
    gridcolor: '#1f2937',
    zerolinecolor: '#374151'
  },
  yaxis: {
    gridcolor: '#1f2937',
    zerolinecolor: '#374151'
  }
};
```

**Dynamic Class Application:**
```javascript
// Apply classification-specific styling
function displayResults(results) {
  const predictionClass = document.getElementById('prediction-class');
  predictionClass.textContent = results.class_name;

  // Add dynamic class based on result
  predictionClass.classList.remove('normal', 'arrhythmia');
  predictionClass.classList.add(
    results.class_name.toLowerCase() === 'normal' ? 'normal' : 'arrhythmia'
  );
}
```

**Animation Triggers:**
```javascript
// Trigger spike animation on new data
function plotSpikes(spikeData) {
  Plotly.newPlot('spike-plot', [spikeTrace], spikeLayout);

  // Add animation class to card
  const card = document.querySelector('.spike-monitor-card');
  card.classList.add('data-updated');
  setTimeout(() => card.classList.remove('data-updated'), 600);
}
```

---

## Performance Considerations

### Bundle Size Optimization

**Current Dependencies:**
- Plotly.js CDN: ~3 MB (largest)
- Custom CSS: ~40 KB (with new styles)
- Custom JS: ~15 KB
- Font files: ~150 KB (3 fonts × 2-3 weights)
- **Total:** ~3.2 MB

**Optimization Strategies:**
1. Use Plotly Basic bundle (1.2 MB) instead of full (3 MB)
2. Font subsetting (include only Latin characters)
3. CSS minification (production build)
4. Enable gzip compression on Flask

**Expected Improvement:**
- Bundle size: 3.2 MB → 1.8 MB (~44% reduction)
- Load time (3G): ~8s → ~4s

### Rendering Performance

**Target Metrics:**
- Page load: < 2s (desktop), < 4s (mobile 3G)
- Time to Interactive (TTI): < 3s
- First Contentful Paint (FCP): < 1.5s
- Animation: 60 FPS (16.67ms per frame)

**Optimization Techniques:**
1. Lazy load Plotly (defer until first interaction)
2. Use `content-visibility: auto` for off-screen cards
3. Debounce resize handlers
4. Use `requestAnimationFrame` for animations
5. Minimize DOM manipulations (batch updates)

**Performance Monitoring:**
```javascript
// Track inference time
performance.mark('inference-start');
await fetch('/api/predict', {...});
performance.mark('inference-end');
performance.measure('inference', 'inference-start', 'inference-end');

const measure = performance.getEntriesByName('inference')[0];
console.log(`Inference took ${measure.duration.toFixed(2)}ms`);
```

### Memory Management

**Plotly Plot Cleanup:**
```javascript
function clearPlot(plotId) {
  Plotly.purge(plotId);  // Free memory before re-plotting
  Plotly.newPlot(plotId, traces, layout);
}
```

**Event Listener Cleanup:**
```javascript
// Use AbortController for cleanup
const controller = new AbortController();
button.addEventListener('click', handler, { signal: controller.signal });

// Later: cleanup
controller.abort();
```

---

## Browser Compatibility

### Minimum Supported Versions

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 90+ | Full support |
| Firefox | 88+ | Full support |
| Safari | 14+ | Full support, test backdrop-filter |
| Edge | 90+ | Full support (Chromium-based) |
| Mobile Safari | 14+ | Test touch interactions |
| Chrome Android | 90+ | Test mobile responsiveness |

### Feature Detection & Fallbacks

**CSS Grid:**
```css
@supports not (display: grid) {
  .status-grid-enhanced {
    display: flex;
    flex-wrap: wrap;
  }
}
```

**Backdrop Filter:**
```css
.card {
  backdrop-filter: blur(10px);

  @supports not (backdrop-filter: blur(10px)) {
    background: rgba(15, 20, 25, 1); /* Opaque fallback */
  }
}
```

**CSS Custom Properties:**
```css
.neural-title {
  color: #00d9ff; /* Fallback */
  color: var(--neural-cyan); /* Modern */
}
```

---

## Testing Checklist

### Visual Regression Testing

- [ ] Header: Title, pulse icon, subtitle visible
- [ ] Status panel: 3 metrics with indicators
- [ ] Controls: Dropdown, 2 buttons, preview box
- [ ] ECG plot: Dark theme, correct colors, grid visible
- [ ] Spike raster: Cyan markers, legend overlay, hero styling
- [ ] Results: Classification large, confidence shown, bars animated
- [ ] Performance metrics: 3 boxes, comparison bars, values updated
- [ ] Footer: Tech badges, attribution text

### Functional Testing

- [ ] Generate normal ECG → plots update
- [ ] Generate arrhythmia ECG → plots update with red color
- [ ] Run prediction → results card appears with correct data
- [ ] Hover buttons → ripple effect + lift animation
- [ ] Page load → cards slide up with stagger
- [ ] Resize window → responsive breakpoints work
- [ ] Mobile view → single column, readable text
- [ ] Keyboard navigation → all interactive elements accessible

### Performance Testing

- [ ] Lighthouse score: 90+ (performance, accessibility, best practices)
- [ ] Animation FPS: 60 (Chrome DevTools Performance tab)
- [ ] Memory: No leaks after 10 generate/predict cycles
- [ ] Bundle size: < 2 MB gzipped
- [ ] Load time (3G): < 4s

### Accessibility Testing

- [ ] Color contrast: WCAG AA (4.5:1 minimum)
- [ ] Keyboard navigation: All buttons/links reachable
- [ ] Screen reader: NVDA/VoiceOver can read all content
- [ ] Focus indicators: Visible on all interactive elements
- [ ] Motion: Respects `prefers-reduced-motion`

### Cross-Browser Testing

- [ ] Chrome 90+: Full functionality
- [ ] Firefox 88+: Full functionality
- [ ] Safari 14+: Backdrop-filter fallback if needed
- [ ] Edge 90+: Full functionality
- [ ] Mobile Safari: Touch interactions work
- [ ] Chrome Android: Responsive layout correct

---

## Future Enhancements (Post-MVP)

### Phase 5: Advanced Features (Days 8-14)

**Real-Time Spike Animation:**
- WebGL-accelerated particle system for spike visualization
- 3D neural network topology view
- Interactive neuron exploration (click to see connections)

**Interactive Parameter Controls:**
- Sliders for spike encoding parameters (gain, num_steps)
- Real-time SNN parameter adjustment (beta, threshold)
- A/B comparison mode (compare two signals side-by-side)

**Enhanced Data Visualization:**
- Membrane potential traces for individual neurons
- Synaptic weight heatmaps (for STDP models)
- Energy consumption live graph (Joules over time)

**Clinical Integration:**
- Upload custom ECG files (CSV/EDF formats)
- Multi-lead ECG support (12-lead visualization)
- Export diagnostic reports (PDF generation)

### Phase 6: Mobile App (Days 15-30)

**Progressive Web App (PWA):**
- Service worker for offline support
- Add to home screen functionality
- Push notifications for inference completion

**Touch Optimizations:**
- Swipe gestures for navigation
- Pinch-to-zoom on plots
- Touch-friendly button sizing (48px minimum)

**Mobile-Specific Features:**
- Camera integration (scan ECG printouts)
- Accelerometer-based Easter eggs (shake to randomize)
- Haptic feedback on interactions

---

## Appendix: Code Snippets

### Neural Background SVG Pattern

Save as `demo/static/neural-bg.svg`:

```svg
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" opacity="0.03">
  <defs>
    <pattern id="neuralNet" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
      <!-- Neurons -->
      <circle cx="10" cy="10" r="2" fill="#00d9ff" />
      <circle cx="90" cy="30" r="2" fill="#00d9ff" />
      <circle cx="50" cy="70" r="2" fill="#00d9ff" />
      <circle cx="30" cy="90" r="2" fill="#00d9ff" />

      <!-- Synapses -->
      <line x1="10" y1="10" x2="90" y2="30" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="90" y1="30" x2="50" y2="70" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="50" y1="70" x2="30" y2="90" stroke="#00d9ff" stroke-width="0.5" opacity="0.3" />
      <line x1="10" y1="10" x2="30" y2="90" stroke="#00d9ff" stroke-width="0.5" opacity="0.2" />
    </pattern>
  </defs>
  <rect width="100%" height="100%" fill="url(#neuralNet)" />
</svg>
```

### Loading Neuron Animation (HTML)

```html
<div class="loading-animation">
  <span class="loading-neuron"></span>
  <span class="loading-neuron"></span>
  <span class="loading-neuron"></span>
</div>
```

### Condition Preview Update (JavaScript)

```javascript
document.getElementById('condition-select').addEventListener('change', (e) => {
  const condition = e.target.value;
  const preview = document.getElementById('condition-preview');

  if (condition === 'normal') {
    preview.textContent = '70 BPM, Low Noise';
  } else {
    preview.textContent = '120 BPM, High Noise';
  }
});
```

---

## Design Credits & Inspiration

**Typography:**
- JetBrains Mono: JetBrains (Open Source)
- Archivo: Omnibus-Type (SIL Open Font License)
- Rajdhani: Indian Type Foundry (OFL)

**Color Palette Inspiration:**
- Neural visualization tools (NEURON, Brian2)
- Medical imaging displays (GE, Siemens)
- IDE dark themes (One Dark Pro, Monokai)
- Cyberpunk aesthetics (Ghost in the Shell, Blade Runner 2049)

**Animation References:**
- Apple Human Interface Guidelines
- Material Design Motion
- Stripe website interactions
- Linear app interface

---

## Conclusion

This redesign transforms CortexCore from a generic "AI slop" interface into a **distinctive, context-aware neuromorphic computing interface** that:

✅ **Avoids generic patterns** (no purple gradients, system fonts, or cookie-cutter layouts)
✅ **Embraces dark neuroscience aesthetic** (brain-inspired, cyberpunk, clinical precision)
✅ **Features minimal but impactful elements** (every component serves a purpose)
✅ **Prioritizes spike visualization** (hero element with dramatic styling)
✅ **Emphasizes clinical accuracy** (diagnostic dashboard, confidence metrics)
✅ **Delivers dynamic animations** (staggered reveals, spike firing, ECG traces)
✅ **Maintains technical credibility** (research-focused, data-dense, monospace fonts)

**Total Implementation Time:** ~20-25 hours across 5 days
**Expected Outcome:** A production-ready, visually stunning demo that stands out in hackathons, research presentations, and technical portfolios.

---

**Document Version:** 2.0
**Last Updated:** 2025-11-07
**Maintainer:** CortexCore Team
**License:** Project-specific (see LICENSE)
