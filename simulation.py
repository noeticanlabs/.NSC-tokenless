import numpy as np
from flask import Flask, send_file, request, jsonify, Response
from scipy.integrate import odeint
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib
import io

app = Flask(__name__)

# --- RFT v3.3 Simulation: Constants & Configuration ---
# -- Model Selection --
SIMULATION_MODEL = 'rft_lattice'

# -- Kuramoto (0D) Parameters --
N_OSCILLATORS = 100
np.random.seed(42)
GAMMA = 0.1
OMEGA = np.random.standard_cauchy(N_OSCILLATORS) * GAMMA
THETA_0 = np.random.uniform(0, 2*np.pi, N_OSCILLATORS)
T_START, T_END, T_STEPS = 0, 100, 1000
T = np.linspace(T_START, T_END, T_STEPS)

# -- RFT Lattice (2D) Parameters --
GRID_SIZE = 50

# --- Vectorized Kuramoto Model ---
def kuramoto_vectorized(theta, t, N_osc, K, omega, D):
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    coupling_sum = sin_theta * np.sum(cos_theta) - cos_theta * np.sum(sin_theta)
    noise = np.random.normal(0, np.sqrt(2 * D), N_osc) if D > 0 else 0
    dtheta_dt = omega + (K/N_osc) * coupling_sum + noise
    return dtheta_dt

# --- RFT 2D Lattice Solver (v3.3 Foundations) ---
class RFTLatticeSolver:
    def __init__(self, size, k1=1.0, alpha=0.1, dt=0.05, glyph_intensity=0.0):
        self.size = size
        self.k1 = k1
        self.alpha = alpha
        self.dt = dt
        self.glyph_intensity = glyph_intensity

        self.theta = np.random.uniform(0, 2 * np.pi, (size, size))
        # FIX: Initialize A with small random values to make the alpha parameter effective.
        self.A = np.random.normal(0, 0.1, (size, size, 2))
        self.glyph_pattern = self._create_glyph_pattern()

    def _create_glyph_pattern(self):
        """Creates a cross-shaped glyph pattern to be imprinted on the field."""
        glyph = np.zeros((self.size, self.size))
        center = self.size // 2
        line_width = max(1, self.size // 20)
        glyph[center-line_width:center+line_width, :] = 1.0
        glyph[:, center-line_width:center+line_width] = 1.0
        return glyph

    def _potential_term(self, theta):
        return np.sin(theta)

    def _axion_term(self):
        dAx_dy = (np.roll(self.A[:,:,0], -1, axis=0) - np.roll(self.A[:,:,0], 1, axis=0)) / 2.0
        dAy_dx = (np.roll(self.A[:,:,1], -1, axis=1) - np.roll(self.A[:,:,1], 1, axis=1)) / 2.0
        F_xy = dAy_dx - dAx_dy
        return (self.alpha / (8 * np.pi)) * F_xy**2

    def step(self):
        grad_theta_x = (np.roll(self.theta, -1, axis=1) - np.roll(self.theta, 1, axis=1)) / 2.0
        grad_theta_y = (np.roll(self.theta, -1, axis=0) - np.roll(self.theta, 1, axis=0)) / 2.0
        
        D_theta_x = grad_theta_x - self.A[:,:,1]
        D_theta_y = grad_theta_y - self.A[:,:,0]

        div_J_theta_x = (np.roll(D_theta_x, 1, axis=1) - np.roll(D_theta_x, -1, axis=1)) / 2.0
        div_J_theta_y = (np.roll(D_theta_y, 1, axis=0) - np.roll(D_theta_y, -1, axis=0)) / 2.0
        
        stiffness_term = self.k1 * (div_J_theta_x + div_J_theta_y)

        # Add the glyph term to the dynamics
        glyph_term = self.glyph_intensity * self.glyph_pattern

        dtheta_dt = -stiffness_term - self._potential_term(self.theta) + self._axion_term() + glyph_term
        
        self.theta += dtheta_dt * self.dt

    def run(self, steps=200):
        for _ in range(steps):
            self.step()
        return self.theta

@app.route('/')
def index():
    html = """
    <html>
        <head>
            <title>RFT Live Simulation Dashboard</title>
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; text-align: center; background-color: #f4f6f8; margin: 0; padding: 2em; }
                h1 { color: #333; }
                #dashboard { background-color: white; max-width: 1000px; margin: auto; padding: 2em; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
                #controls { display: flex; justify-content: center; align-items: center; gap: 2rem; margin-bottom: 2em; flex-wrap: wrap; }
                .control { display: flex; flex-direction: column; align-items: center; }
                .control label { font-weight: bold; margin-bottom: 0.5em; color: #555; }
                .control input[type=range], .control select { width: 200px; }
                .control input:disabled + label { color: #ccc; }
                img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 1em; }
            </style>
        </head>
        <body>
            <div id="dashboard">
                <h1>RFT Live Simulation Dashboard</h1>
                <div id="controls">
                    <div class="control">
                        <label for="model_select">Model</label>
                        <select id="model_select">
                            <option value="rft_lattice" selected>RFT Lattice (2D)</option>
                            <option value="kuramoto">Kuramoto (0D)</option>
                        </select>
                    </div>
                    <div class="control">
                        <label for="k_slider">Coupling (K): <span id="k_value">2.0</span></label>
                        <input type="range" id="k_slider" min="0" max="10" step="0.1" value="2.0">
                    </div>
                    <div class="control">
                        <label for="d_slider">Noise (D): <span id="d_value">0.0</span></label>
                        <input type="range" id="d_slider" min="0" max="1" step="0.01" value="0.0">
                    </div>
                    <div class="control">
                        <label for="alpha_slider">Alpha: <span id="alpha_value">0.1</span></label>
                        <input type="range" id="alpha_slider" min="0" max="1" step="0.01" value="0.1">
                    </div>
                    <div class="control">
                        <label for="glyph_slider">Glyph Intensity: <span id="glyph_value">0.0</span></label>
                        <input type="range" id="glyph_slider" min="0" max="10" step="0.1" value="0.0">
                    </div>
                </div>
                <img id="plot" src="" alt="Loading simulation plot..."/>
            </div>

            <script>
                const modelSelect = document.getElementById('model_select');
                const kSlider = document.getElementById('k_slider');
                const dSlider = document.getElementById('d_slider');
                const alphaSlider = document.getElementById('alpha_slider');
                const glyphSlider = document.getElementById('glyph_slider');
                const kValueSpan = document.getElementById('k_value');
                const dValueSpan = document.getElementById('d_value');
                const alphaValueSpan = document.getElementById('alpha_value');
                const glyphValueSpan = document.getElementById('glyph_value');
                const plotImage = document.getElementById('plot');

                function updateControlStates() {
                    const model = modelSelect.value;
                    if (model === 'kuramoto') {
                        dSlider.disabled = false;
                        alphaSlider.disabled = true;
                        glyphSlider.disabled = true;
                    } else { // rft_lattice
                        dSlider.disabled = true;
                        alphaSlider.disabled = false;
                        glyphSlider.disabled = false;
                    }
                }

                function updatePlot() {
                    const model = modelSelect.value;
                    const k = kSlider.value;
                    const d = dSlider.value;
                    const alpha = alphaSlider.value;
                    const glyph = glyphSlider.value;
                    kValueSpan.textContent = k;
                    dValueSpan.textContent = d;
                    alphaValueSpan.textContent = alpha;
                    glyphValueSpan.textContent = glyph;
                    
                    const timestamp = new Date().getTime();
                    plotImage.src = `/plot/${model}.png?k=${k}&d=${d}&alpha=${alpha}&glyph=${glyph}&t=${timestamp}`;
                }

                modelSelect.addEventListener('input', () => {
                    updateControlStates();
                    updatePlot();
                });
                kSlider.addEventListener('input', updatePlot);
                dSlider.addEventListener('input', updatePlot);
                alphaSlider.addEventListener('input', updatePlot);
                glyphSlider.addEventListener('input', updatePlot);

                document.addEventListener('DOMContentLoaded', () => {
                    updateControlStates();
                    updatePlot();
                });
            </script>
        </body>
    </html>
    """
    return Response(html)

@app.route('/plot.png', defaults={'model': SIMULATION_MODEL})
@app.route('/plot/<string:model>.png')
def plot_png(model):
    K = float(request.args.get('k', default=2.0))
    D = float(request.args.get('d', default=0.0))
    alpha = float(request.args.get('alpha', default=0.1))
    glyph = float(request.args.get('glyph', default=0.0))
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 13))
    fig.suptitle(f'RFT Analysis (model={model}, K={K}, D={D}, alpha={alpha}, glyph={glyph})', fontsize=16)

    if model == 'kuramoto':
        theta_t = odeint(kuramoto_vectorized, THETA_0, T, args=(N_OSCILLATORS, K, OMEGA, D))
        R = np.abs(np.mean(np.exp(1j * theta_t), axis=1))

        ax1.plot(T, np.unwrap(theta_t[:, 0]), label='Oscillator 1')
        ax1.plot(T, np.unwrap(theta_t[:, N_OSCILLATORS//2]), label=f'Oscillator {N_OSCILLATORS//2}')
        ax1.set_title('Individual Oscillator Dynamics')
        ax1.legend()
        
        ax2.plot(T, R)
        ax2.set_title('Overall Phase Coherence (R)')
        ax2.set_ylim([0, 1])

        steady_state_index = T_STEPS // 2
        R_steady = R[steady_state_index:]
        if len(R_steady) > 0:
            yf = fft(R_steady - np.mean(R_steady))
            xf = fftfreq(len(R_steady), (T_END - T_START) / T_STEPS)
            power = np.abs(yf)**2
            ax3.plot(xf[:len(xf)//2], power[:len(power)//2])
        ax3.set_title('Power Spectrum of Collective Rhythm')

        ax4.text(0.5, 0.5, "Kuramoto Model Selected", ha='center', va='center')
        ax4.set_title('2D Lattice Field State')

    elif model == 'rft_lattice':
        rft_solver = RFTLatticeSolver(size=GRID_SIZE, k1=K, alpha=alpha, glyph_intensity=glyph)
        final_lattice_state = rft_solver.run(steps=200)

        ax1.text(0.5, 0.5, "RFT Lattice Model Selected", ha='center', va='center')
        ax1.set_title('Individual Oscillator Dynamics')
        ax2.text(0.5, 0.5, "RFT Lattice Model Selected", ha='center', va='center')
        ax2.set_title('Overall Phase Coherence (R)')
        ax3.text(0.5, 0.5, "RFT Lattice Model Selected", ha='center', va='center')
        ax3.set_title('Power Spectrum of Collective Rhythm')

        im = ax4.imshow(np.sin(final_lattice_state), cmap='twilight_shifted', interpolation='bilinear')
        ax4.set_title('2D Lattice Field State (sin(θ))')
        fig.colorbar(im, ax=ax4, orientation='horizontal', fraction=0.046, pad=0.04)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.grid(True)
        if not ax.get_title(): ax.set_title("Inactive Plot")
        if not ax.get_xlabel(): ax.set_xlabel("Time" if ax in [ax1, ax2] else "Frequency (Hz)")
        if not ax.get_ylabel(): ax.set_ylabel("Phase" if ax == ax1 else "Coherence" if ax == ax2 else "Power")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    return send_file(buf, mimetype='image/png')
