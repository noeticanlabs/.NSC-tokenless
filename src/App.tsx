import React, { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "./components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select";
import { Switch } from "./components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Play, Pause, RotateCcw, Download, Activity, Grid as GridIcon, Crosshair, Layers } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend, ResponsiveContainer } from "recharts";

/**
 * RFT Multi‑Layer Web Simulation (v1)
 *
 * A real‑time, in‑browser verification harness for the Resonant Field Theory (RFT)
 * scalar–gauge sector using a 2D XY phase field on a torus with optional background holonomy.
 *
 * Layers:
 *  - Phase field (HSV color wheel)
 *  - Vorticity markers (±1 vortices)
 *  - Gauge field arrows (A_x, A_y) for holonomy visualization
 *  - Metrics plots: R(t), E_xy(t), Q_total(t), separation & braid phase Φ(t)
 *
 * Controls:
 *  - Lattice size N, time step dt
 *  - Noise σ
 *  - Correction cadence τ_c and step η
 *  - Coupling J
 *  - Holonomy (n_x, n_y)
 *  - Deterministic seed
 *
 * CSV export for timeseries; deterministic RNG; pause/run/reset.
 */

// ---------- Utility: deterministic RNG (Mulberry32) ----------
function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// wrap angle to (-π, π]
function wrap(a: number) {
  const PI = Math.PI;
  const TWO_PI = 2 * PI;
  a = ((a + PI) % (TWO_PI)) - PI;
  if (a <= -PI) a += TWO_PI;
  return a;
}

// HSV to RGB for phase visualization
function hsvToRgb(h: number, s: number, v: number) {
  let f = (n: number) => {
    let k = (n + h * 6) % 6;
    return v - v * s * Math.max(Math.min(k, 4 - k, 1), 0);
  };
  return [f(5), f(3), f(1)];
}

// ---------- Core Simulation ----------
interface Params {
  N: number;
  dt: number; // timestep for visualization pacing
  sigma: number; // noise std dev per step
  tauC: number; // correction cadence (steps)
  eta: number; // correction step size
  J: number; // XY coupling strength
  nx: number; // holonomy integer along x
  ny: number; // holonomy integer along y
  showPhase: boolean;
  showVortices: boolean;
  showGauge: boolean;
  seed: number;
}

interface Timesample {
  t: number;
  R: number;
  Exy: number;
  Q: number;
  Nplus: number;
  Nminus: number;
  sep: number | null;
  Phi: number;
}

function useRFTSim(initial: Params) {
  const [params, setParams] = useState<Params>(initial);
  const [running, setRunning] = useState(false);
  const [t, setT] = useState(0);
  const randRef = useRef<() => number>(mulberry32(initial.seed));

  // fields
  const thetaRef = useRef<Float32Array | null>(null);
  const AxRef = useRef<Float32Array | null>(null);
  const AyRef = useRef<Float32Array | null>(null);

  // diagnostics
  const [series, setSeries] = useState<Timesample[]>([]);
  const phiRef = useRef(0); // braid phase
  const prevAngleRef = useRef<number | null>(null);

  // initialize lattice
  const reset = (p = params) => {
    const N = p.N;
    thetaRef.current = new Float32Array(N * N);
    AxRef.current = new Float32Array(N * N);
    AyRef.current = new Float32Array(N * N);
    // Initial condition: two vortices (+1, -1) separated horizontally
    const cx1 = Math.floor(N * 0.4);
    const cy1 = Math.floor(N * 0.5);
    const cx2 = Math.floor(N * 0.6);
    const cy2 = Math.floor(N * 0.5);
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const dx1 = ((x - cx1 + N + N / 2) % N) - N / 2;
        const dy1 = ((y - cy1 + N + N / 2) % N) - N / 2;
        const dx2 = ((x - cx2 + N + N / 2) % N) - N / 2;
        const dy2 = ((y - cy2 + N + N / 2) % N) - N / 2;
        const a1 = Math.atan2(dy1, dx1);
        const a2 = Math.atan2(dy2, dx2);
        const idx = y * N + x;
        thetaRef.current![idx] = wrap(a1 - a2); // +1 and -1
      }
    }
    // background gauge for holonomy: Ax = 2π nx / N, Ay = 2π ny / N (uniform)
    const Ax0 = (2 * Math.PI * p.nx) / N;
    const Ay0 = (2 * Math.PI * p.ny) / N;
    AxRef.current.fill(Ax0);
    AyRef.current.fill(Ay0);

    randRef.current = mulberry32(p.seed);
    setT(0);
    phiRef.current = 0;
    prevAngleRef.current = null;
    setSeries([]);
  };

  useEffect(() => {
    reset(params);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // discrete gradient and energy/coherence diagnostics
  const stepOnce = () => {
    const p = params;
    const N = p.N;
    const theta = thetaRef.current!;
    const Ax = AxRef.current!;
    const Ay = AyRef.current!;
    const rand = randRef.current;

    // 1) Noise kick
    for (let i = 0; i < theta.length; i++) {
      // Box-Muller
      const u1 = Math.max(1e-12, rand());
      const u2 = Math.max(1e-12, rand());
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      theta[i] = wrap(theta[i] + p.sigma * z);
    }

    // 2) Optional correction step (gradient descent) every tauC steps
    if ((t + 1) % p.tauC === 0) {
      const newTheta = new Float32Array(theta.length);
      for (let y = 0; y < N; y++) {
        for (let x = 0; x < N; x++) {
          const i = y * N + x;
          // 4-nn neighbors with periodic BC
          const xm = (x - 1 + N) % N;
          const xp = (x + 1) % N;
          const ym = (y - 1 + N) % N;
          const yp = (y + 1) % N;
          const i_xm = y * N + xm;
          const i_xp = y * N + xp;
          const i_ym = ym * N + x;
          const i_yp = yp * N + x;
          // gauge-covariant differences
          const dtxm = wrap(theta[i_xm] - theta[i] - Ax[i]);
          const dtxp = wrap(theta[i_xp] - theta[i] + Ax[i]);
          const dtym = wrap(theta[i_ym] - theta[i] - Ay[i]);
          const dtyp = wrap(theta[i_yp] - theta[i] + Ay[i]);
          const grad = dtxm + dtxp + dtym + dtyp; // Laplacian-like
          newTheta[i] = wrap(theta[i] + p.eta * p.J * grad);
        }
      }
      theta.set(newTheta);
    }

    // 3) Diagnostics: coherence R, energy Exy, vorticity counts, Q_total
    let cx = 0, cy = 0; // mean phasor
    for (let i = 0; i < theta.length; i++) {
      cx += Math.cos(theta[i]);
      cy += Math.sin(theta[i]);
    }
    const R = Math.sqrt(cx * cx + cy * cy) / (N * N);

    // Energy (gauge-covariant XY)
    let E = 0;
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const xp = (x + 1) % N;
        const yp = (y + 1) % N;
        const ixp = y * N + xp;
        const iyp = yp * N + x;
        const dx = wrap(theta[ixp] - theta[i] - Ax[i]);
        const dy = wrap(theta[iyp] - theta[i] - Ay[i]);
        E += -(p.J) * (Math.cos(dx) + Math.cos(dy));
      }
    }
    const Exy = E / (N * N);

    // Vorticity via plaquette sum
    let Q = 0, Nplus = 0, Nminus = 0;
    let posPlus: [number, number][] = [];
    let posMinus: [number, number][] = [];
    for (let y = 0; y < N; y++) {
      for (let x = 0; x < N; x++) {
        const i = y * N + x;
        const xp = (x + 1) % N;
        const yp = (y + 1) % N;
        const ixp = y * N + xp;
        const iyp = yp * N + x;
        const ixpyp = yp * N + xp;
        const e1 = wrap(theta[ixp] - theta[i] - Ax[i]);
        const e2 = wrap(theta[ixpyp] - theta[ixp] - Ay[ixp]);
        const e3 = wrap(theta[iyp] - theta[ixpyp] + Ax[iyp]);
        const e4 = wrap(theta[i] - theta[iyp] + Ay[i]);
        const q = (e1 + e2 + e3 + e4) / (2 * Math.PI);
        const qi = Math.round(q);
        if (qi === 1) { Nplus++; Q++; posPlus.push([x + 0.5, y + 0.5]); }
        if (qi === -1) { Nminus++; Q--; posMinus.push([x + 0.5, y + 0.5]); }
      }
    }

    // Track braid phase using first + and - if available
    let sep: number | null = null;
    if (posPlus.length && posMinus.length) {
      // nearest pair
      const p1 = posPlus[0];
      let best = posMinus[0];
      let bestd = 1e9;
      for (const m of posMinus) {
        const dx = ((m[0] - p1[0] + N + N / 2) % N) - N / 2;
        const dy = ((m[1] - p1[1] + N + N / 2) % N) - N / 2;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestd) { bestd = d2; best = m; }
      }
      const dx = ((best[0] - p1[0] + N + N / 2) % N) - N / 2;
      const dy = ((best[1] - p1[1] + N + N / 2) % N) - N / 2;
      sep = Math.sqrt(bestd);
      const angle = Math.atan2(dy, dx);
      if (prevAngleRef.current == null) prevAngleRef.current = angle;
      let dphi = wrap(angle - (prevAngleRef.current as number));
      phiRef.current += dphi;
      prevAngleRef.current = angle;
    }

    const sample: Timesample = {
      t: t + 1,
      R,
      Exy,
      Q,
      Nplus,
      Nminus,
      sep,
      Phi: phiRef.current,
    };
    setSeries((s) => (s.length > 2000 ? [...s.slice(-1000), sample] : [...s, sample]));
    setT((tt) => tt + 1);
  };

  useEffect(() => {
    if (!running) return;
    let raf: number;
    let last = performance.now();
    const loop = () => {
      const now = performance.now();
      if (now - last > params.dt) {
        stepOnce();
        last = now;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, params]);

  return {
    params,
    setParams,
    running,
    setRunning,
    t,
    series,
    thetaRef,
    AxRef,
    AyRef,
    reset,
    stepOnce,
  };
}

// ---------- Visualization Components ----------
function PhaseCanvas({ sim }: { sim: ReturnType<typeof useRFTSim> }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const { params, thetaRef, AxRef, AyRef } = sim;

  useEffect(() => {
    const N = params.N;
    const theta = thetaRef.current;
    const Ax = AxRef.current;
    const Ay = AyRef.current;
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !theta) return;

    const scale = Math.max(2, Math.floor(512 / N));
    canvasRef.current!.width = N * scale;
    canvasRef.current!.height = N * scale;
    const img = ctx.createImageData(N, N);
    for (let i = 0; i < theta.length; i++) {
      const ph = (theta[i] + Math.PI) / (2 * Math.PI);
      const [r, g, b] = hsvToRgb(ph, 1, 1);
      img.data[4 * i + 0] = Math.floor(r * 255);
      img.data[4 * i + 1] = Math.floor(g * 255);
      img.data[4 * i + 2] = Math.floor(b * 255);
      img.data[4 * i + 3] = 255;
    }
    // put then scale up
    const off = document.createElement("canvas");
    off.width = N; off.height = N;
    const octx = off.getContext("2d")!;
    octx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, N * scale, N * scale);

    // overlay gauge arrows if enabled
    if (params.showGauge && Ax && Ay) {
      ctx.strokeStyle = "rgba(255,255,255,0.5)";
      const step = Math.max(1, Math.floor(N / 16));
      for (let y = 0; y < N; y += step) {
        for (let x = 0; x < N; x += step) {
          const i = y * N + x;
          const ax = Ax[i];
          const ay = Ay[i];
          const cx = (x + 0.5) * scale;
          const cy = (y + 0.5) * scale;
          const len = Math.min(8, 0.5 * scale * Math.hypot(ax, ay));
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx + len * Math.cos(0), cy);
          ctx.stroke();
          // small cross to indicate Ay upward influence
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(cx, cy - len);
          ctx.stroke();
        }
      }
    }

    // overlay vortices if enabled: sample plaquettes coarsely
    if (params.showVortices && theta) {
      const N = params.N;
      const stepV = Math.max(1, Math.floor(N / 32));
      for (let y = 0; y < N; y += stepV) {
        for (let x = 0; x < N; x += stepV) {
          const i = y * N + x;
          const xp = (x + 1) % N;
          const yp = (y + 1) % N;
          const ixp = y * N + xp;
          const iyp = yp * N + x;
          const ixpyp = yp * N + xp;
          const e1 = wrap(theta[ixp] - theta[i] - (AxRef.current?.[i] ?? 0));
          const e2 = wrap(theta[ixpyp] - theta[ixp] - (AyRef.current?.[ixp] ?? 0));
          const e3 = wrap(theta[iyp] - theta[ixpyp] + (AxRef.current?.[iyp] ?? 0));
          const e4 = wrap(theta[i] - theta[iyp] + (AyRef.current?.[i] ?? 0));
          const q = Math.round((e1 + e2 + e3 + e4) / (2 * Math.PI));
          if (q !== 0) {
            const ctx2 = ctx;
            ctx2.beginPath();
            ctx2.arc((x + 0.5) * scale, (y + 0.5) * scale, 4, 0, 2 * Math.PI);
            ctx2.fillStyle = q > 0 ? "#00ff88" : "#ff3366";
            ctx2.fill();
          }
        }
      }
    }
  }, [sim.t, params.showGauge, params.showVortices, params.N]);

  return (
    <div className="rounded-xl overflow-hidden border shadow-sm">
      <canvas ref={canvasRef} className="block w-full h-auto bg-black" />
    </div>
  );
}

function MetricsPanel({ series }: { series: Timesample[] }) {
  const latest = series[series.length - 1];
  const data = series.slice(-400).map((d) => ({
    t: d.t,
    R: d.R,
    Exy: d.Exy,
    Q: d.Q,
    Phi: d.Phi,
    sep: d.sep ?? 0,
  }));
  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Activity className="h-5 w-5"/> Live Metrics</CardTitle>
      </CardHeader>
      <CardContent>
        {latest && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4 text-sm">
            <div className="p-3 rounded-lg bg-muted">R: <span className="font-mono">{latest.R.toFixed(3)}</span></div>
            <div className="p-3 rounded-lg bg-muted">E_xy: <span className="font-mono">{latest.Exy.toFixed(4)}</span></div>
            <div className="p-3 rounded-lg bg-muted">Q_total: <span className="font-mono">{latest.Q}</span></div>
            <div className="p-3 rounded-lg bg-muted">Φ: <span className="font-mono">{latest.Phi.toFixed(3)}</span></div>
          </div>
        )}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" />
                <YAxis yAxisId="left" />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="R" dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="Exy" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data} margin={{ left: 10, right: 10, top: 10, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="t" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="Q" dot={false} />
                <Line type="monotone" dataKey="Phi" dot={false} />
                <Line type="monotone" dataKey="sep" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function Controls({ sim }: { sim: ReturnType<typeof useRFTSim> }) {
  const { params, setParams, reset, running, setRunning, stepOnce } = sim;

  const update = (patch: Partial<Params>) => setParams({ ...params, ...patch });

  // CSV export
  const downloadCSV = () => {
    const header = ["t", "R", "E_xy", "Q_total", "N_plus", "N_minus", "sep", "Phi"].join(",");
    const rows = sim.series.map((d) => [d.t, d.R, d.Exy, d.Q, d.Nplus, d.Nminus, d.sep ?? "", d.Phi].join(","));
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "RFT_verification_timeseries.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2"><Layers className="h-5 w-5"/> Controls</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div>
            <Label>N</Label>
            <Input type="number" value={params.N} onChange={(e) => update({ N: Math.max(16, Math.min(256, parseInt(e.target.value||"64"))) })} />
          </div>
          <div>
            <Label>σ (noise)</Label>
            <Input type="number" step="0.005" value={params.sigma} onChange={(e) => update({ sigma: parseFloat(e.target.value||"0") })} />
          </div>
          <div>
            <Label>τ_c (steps)</Label>
            <Input type="number" value={params.tauC} onChange={(e) => update({ tauC: Math.max(1, parseInt(e.target.value||"10")) })} />
          </div>
          <div>
            <Label>η (corr step)</Label>
            <Input type="number" step="0.01" value={params.eta} onChange={(e) => update({ eta: parseFloat(e.target.value||"0.2") })} />
          </div>
          <div>
            <Label>J</Label>
            <Input type="number" step="0.1" value={params.J} onChange={(e) => update({ J: parseFloat(e.target.value||"1.0") })} />
          </div>
          <div>
            <Label>Holonomy n_x</Label>
            <Input type="number" value={params.nx} onChange={(e) => update({ nx: parseInt(e.target.value||"0") })} />
          </div>
          <div>
            <Label>Holonomy n_y</Label>
            <Input type="number" value={params.ny} onChange={(e) => update({ ny: parseInt(e.target.value||"0") })} />
          </div>
          <div>
            <Label>Seed</Label>
            <Input type="number" value={params.seed} onChange={(e) => update({ seed: parseInt(e.target.value||"1") })} />
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2"><Switch checked={params.showPhase} onCheckedChange={(v) => update({ showPhase: v })}/><Label>Phase</Label></div>
          <div className="flex items-center gap-2"><Switch checked={params.showVortices} onCheckedChange={(v) => update({ showVortices: v })}/><Label>Vortices</Label></div>
          <div className="flex items-center gap-2"><Switch checked={params.showGauge} onCheckedChange={(v) => update({ showGauge: v })}/><Label>Gauge</Label></div>
        </div>
        <div className="flex flex-wrap gap-3">
          <Button onClick={() => setRunning(!running)} variant={running ? "destructive" : "default"}>
            {running ? <Pause className="h-4 w-4 mr-2"/> : <Play className="h-4 w-4 mr-2"/>}
            {running ? "Pause" : "Run"}
          </Button>
          <Button onClick={() => { setRunning(false); reset(); }} variant="secondary">
            <RotateCcw className="h-4 w-4 mr-2"/> Reset
          </Button>
          <Button onClick={() => stepOnce()} variant="outline">
            Step
          </Button>
          <Button onClick={downloadCSV}>
            <Download className="h-4 w-4 mr-2"/> Export CSV
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

export default function RFTVerificationSim() {
  const sim = useRFTSim({
    N: 96,
    dt: 16,
    sigma: 0.04,
    tauC: 10,
    eta: 0.2,
    J: 1.0,
    nx: 0,
    ny: 0,
    showPhase: true,
    showVortices: true,
    showGauge: false,
    seed: 1337,
  });

  return (
    <div className="p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">RFT Multi‑Layer Web Simulation (v1)</h1>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <PhaseCanvas sim={sim} />
          <MetricsPanel series={sim.series} />
        </div>
        <div className="space-y-6">
          <Controls sim={sim} />
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><GridIcon className="h-5 w-5"/> Guidance</CardTitle>
            </CardHeader>
            <CardContent className="text-sm space-y-2">
              <p><strong>Goal:</strong> verify key RFT laws in‑browser: (i) energy–coherence relation, (ii) topological integrity under noise+correction, (iii) holonomy‑assisted charge, (iv) braid‑phase accumulation.</p>
              <ul className="list-disc pl-5 space-y-1">
                <li>Increase <code>σ</code> to add decoherence; periodic correction (τ_c, η) restores <code>R</code> while conserving charge.</li>
                <li>Set holonomy <code>n_x,n_y ≠ 0</code> to test torus energy scaling vs |n|.</li>
                <li>Watch <code>Φ(t)</code> grow in steps when the +/− pair orbits; <code>sep</code> stays finite.</li>
                <li>Export CSV to compare against offline analysis and published artifacts.</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
