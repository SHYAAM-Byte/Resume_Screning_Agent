import { DatePipe } from '@angular/common';
import { Component, computed, effect, inject, input, signal, untracked } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { ChartData, ChartOptions, Scale } from 'chart.js';
import { BaseChartDirective, provideCharts, withDefaultRegisterables } from 'ng2-charts';
import { Api, CounterSignal, DigitalSignal, Report, errorText } from './api';

type Signal = DigitalSignal | CounterSignal;

interface PlotConfig {
  data: ChartData<'line'>;
  options: ChartOptions<'line'>;
}

/** Y-axis settings that differ between digital and counter plots. */
interface YAxis {
  min?: number;
  max?: number;
  integerTicks?: boolean;
  bitTicks?: boolean;
}

/** What one plot panel shows in its header and footer. */
interface Card {
  key: string;
  sig: Signal;
  canId: string;
  location: string;
  kind: 'digital' | 'counter';
  kindLabel: string;
  detail: string;
  samples: number;
}

const cssVar = (name: string): string =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim();

@Component({
  selector: 'app-report',
  imports: [BaseChartDirective, RouterLink, DatePipe],
  providers: [provideCharts(withDefaultRegisterables())],
  template: `
    @if (error()) {
      <section class="page narrow">
        <p class="alert" role="alert">{{ error() }}</p>
        <a class="btn ghost" routerLink="/process">Back to CSV</a>
      </section>
    } @else if (data(); as r) {
      <section class="page wide">
        <header class="report-head">
          <div>
            <h1>{{ r.project.project_name }}</h1>
            <p class="lede">
              {{ r.project.vehicle_year }} {{ r.project.vehicle_make }} {{ r.project.vehicle_model }}
            </p>
          </div>
          <dl class="facts">
            <div><dt>File</dt><dd>{{ r.project.csv_filename }}</dd></div>
            <div><dt>Rows analysed</dt><dd>{{ r.project.row_count }}</dd></div>
            <div><dt>Processed</dt><dd>{{ r.project.created_at | date: 'medium' }}</dd></div>
          </dl>
        </header>

        @if (r.skipped_rows) {
          <p class="note" role="status">
            {{ r.skipped_rows }} rows were skipped because a time, id or byte value was invalid.
          </p>
        }

        <div class="tabs" role="tablist" aria-label="Signal type">
          <button type="button" role="tab" id="tab-digital" class="tab digital"
                  [attr.aria-selected]="tab() === 'digital'"
                  [attr.tabindex]="tab() === 'digital' ? 0 : -1"
                  (click)="setTab('digital')"
                  (keydown.arrowright)="moveTab('counter')" (keydown.arrowleft)="moveTab('counter')">
            Digital signals <span class="count">{{ r.digital_signals.length }}</span>
          </button>
          <button type="button" role="tab" id="tab-counter" class="tab counter"
                  [attr.aria-selected]="tab() === 'counter'"
                  [attr.tabindex]="tab() === 'counter' ? 0 : -1"
                  (click)="setTab('counter')"
                  (keydown.arrowright)="moveTab('digital')" (keydown.arrowleft)="moveTab('digital')">
            Counter signals <span class="count">{{ r.counter_signals.length }}</span>
          </button>
          <a class="again" routerLink="/process">Process another file</a>
        </div>

        <div role="tabpanel" [attr.aria-labelledby]="'tab-' + tab()">
          <div class="toolbar">
            @if (canIds().length > 1) {
              <label for="idfilter">Show CAN ID</label>
              <select id="idfilter" (change)="onFilter($event)">
                <option value="all" [selected]="idFilter() === 'all'">All ({{ canIds().length }})</option>
                @for (id of canIds(); track id) {
                  <option [value]="id" [selected]="idFilter() === id">{{ id }}</option>
                }
              </select>
            }
            @if (tab() === 'digital') {
              <span class="muted">Bits are numbered 0 (MSB) to 7 (LSB) inside each byte.</span>
            }
          </div>

          @if (cards().length === 0) {
            <p class="empty">
              @if (tab() === 'digital') {
                No digital signals are stored for this project. Either Digital signals was not
                selected, or no bit switched between 0 and 1 and held each state for a few samples.
              } @else {
                No counter signals are stored for this project. Either Counter signals was not
                selected, or no byte counted up by a steady step and wrapped back to its minimum.
              }
            </p>
          } @else {
            <div class="plots">
              @for (c of cards(); track c.key) {
                <article class="plot" [class.counter]="c.kind === 'counter'">
                  <header class="plot-head">
                    <strong class="plot-id">CAN ID {{ c.canId }}</strong>
                    <span class="plot-loc">{{ c.location }}</span>
                    <span class="kind" [class.counter]="c.kind === 'counter'">{{ c.kindLabel }}</span>
                  </header>
                  <div class="plot-body">
                    @defer (on viewport) {
                      <canvas baseChart type="line"
                              [data]="plot(c.sig).data" [options]="plot(c.sig).options"></canvas>
                    } @placeholder {
                      <div class="plot-wait">The chart loads when it scrolls into view.</div>
                    }
                  </div>
                  <footer class="plot-foot">
                    <span>{{ c.detail }}</span>
                    <span>{{ c.samples }} samples</span>
                  </footer>
                </article>
              }
            </div>
          }
        </div>
      </section>
    } @else {
      <p class="muted">Loading report…</p>
    }
  `,
})
export class ReportPage {
  private readonly api = inject(Api);

  /** Route parameter (bound by withComponentInputBinding). */
  readonly id = input.required<string>();

  protected readonly data = signal<Report | null>(null);
  protected readonly error = signal('');
  protected readonly tab = signal<'digital' | 'counter'>('digital');
  protected readonly idFilter = signal('all');

  private readonly active = computed<Signal[]>(() =>
    this.tab() === 'digital'
      ? (this.data()?.digital_signals ?? [])
      : (this.data()?.counter_signals ?? []),
  );
  protected readonly canIds = computed(() => [...new Set(this.active().map((s) => s.can_id))]);
  protected readonly cards = computed<Card[]>(() => {
    const only = this.idFilter();
    return this.active()
      .filter((s) => only === 'all' || s.can_id === only)
      .map((s) =>
        s.signal_type === 'digital'
          ? {
              key: `d-${s.can_id}-${s.byte}-${s.bit}`,
              sig: s,
              canId: s.can_id,
              location: `Byte B${s.byte}, bit ${s.bit}`,
              kind: 'digital' as const,
              kindLabel: 'Digital signal',
              detail: `${s.transitions} state changes`,
              samples: s.abs_time.length,
            }
          : {
              key: `c-${s.can_id}-${s.byte}`,
              sig: s,
              canId: s.can_id,
              location: `Byte B${s.byte}`,
              kind: 'counter' as const,
              kindLabel: 'Counter signal',
              detail: `Counts ${s.min} to ${s.max}, step ${s.step}`,
              samples: s.abs_time.length,
            },
      );
  });

  /** Chart configs are built on first use and cached, so scrolling stays cheap. */
  private readonly plots = new WeakMap<Signal, PlotConfig>();

  constructor() {
    effect(() => {
      const id = Number(this.id());
      const cached = this.api.report();
      untracked(() => {
        if (this.data()?.project.id === id) return;
        if (cached?.project.id === id) this.show(cached);
        else this.fetch(id);
      });
    });
  }

  private show(r: Report): void {
    this.data.set(r);
    this.error.set('');
    this.idFilter.set('all');
    this.tab.set(r.digital_signals.length || !r.counter_signals.length ? 'digital' : 'counter');
  }

  private fetch(id: number): void {
    if (!Number.isInteger(id)) {
      this.error.set('That project could not be found.');
      return;
    }
    this.api.project(id).subscribe({
      next: (r) => {
        this.show(r);
        this.api.report.set(r);
      },
      error: (e) => this.error.set(errorText(e)),
    });
  }

  protected setTab(t: 'digital' | 'counter'): void {
    this.tab.set(t);
    this.idFilter.set('all');
  }

  protected moveTab(t: 'digital' | 'counter'): void {
    this.setTab(t);
    document.getElementById(`tab-${t}`)?.focus();
  }

  protected onFilter(event: Event): void {
    this.idFilter.set((event.target as HTMLSelectElement).value);
  }

  protected plot(s: Signal): PlotConfig {
    let p = this.plots.get(s);
    if (!p) {
      p = s.signal_type === 'digital' ? this.digitalPlot(s) : this.counterPlot(s);
      this.plots.set(s, p);
    }
    return p;
  }

  // ---------------------------------------------------------------- chart configs

  private digitalPlot(s: DigitalSignal): PlotConfig {
    const options = this.baseOptions(`B${s.byte} bit ${s.bit} value`, 'min-max', (v) => `${v}`, {
      min: -0.15,
      max: 1.15,
      bitTicks: true, // only 0 and 1 are meaningful for a bit
    });
    return { data: this.dataset(s, cssVar('--digital'), true), options };
  }

  private counterPlot(s: CounterSignal): PlotConfig {
    const hex = (v: number) => `${v} (0x${v.toString(16).toUpperCase().padStart(2, '0')})`;
    const options = this.baseOptions(`B${s.byte} byte value`, 'lttb', hex, {
      min: Math.max(0, Math.floor(s.min - (s.max - s.min) * 0.08)),
      max: Math.ceil(s.max + (s.max - s.min) * 0.08),
      integerTicks: true,
    });
    return { data: this.dataset(s, cssVar('--counter'), false), options };
  }

  private dataset(s: Signal, color: string, stepped: boolean): ChartData<'line'> {
    return {
      datasets: [
        {
          // {x, y} points + parsing:false is what lets Chart.js decimate large series.
          data: s.abs_time.map((x, i) => ({ x, y: s.values[i] })),
          parsing: false,
          normalized: true,
          borderColor: color,
          backgroundColor: color,
          borderWidth: 1.75,
          pointRadius: s.abs_time.length <= 120 ? 2 : 0,
          pointHoverRadius: 4,
          stepped: stepped ? 'after' : false, // a CAN value holds until the next frame
          tension: 0,
        },
      ],
    };
  }

  private baseOptions(
    yTitle: string,
    algorithm: 'min-max' | 'lttb',
    format: (v: number) => string,
    y: YAxis,
  ): ChartOptions<'line'> {
    const { integerTicks, bitTicks, ...yScale } = y;
    const text = cssVar('--ink-2');
    const grid = cssVar('--line');
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'nearest', axis: 'x', intersect: false },
      plugins: {
        legend: { display: false },
        decimation:
          algorithm === 'lttb'
            ? { enabled: true, algorithm: 'lttb', samples: 600 }
            : { enabled: true, algorithm: 'min-max' },
        tooltip: {
          callbacks: {
            title: (items) => `abs_time ${items[0].parsed.x}`,
            label: (item) => `${yTitle}: ${format(item.parsed.y as number)}`,
          },
        },
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'abs_time', color: text },
          ticks: { color: text, maxTicksLimit: 8 },
          grid: { color: grid },
        },
        y: {
          type: 'linear',
          ...yScale,
          title: { display: true, text: yTitle, color: text },
          ticks: { color: text, ...(integerTicks ? { precision: 0 } : {}) },
          grid: { color: grid },
          ...(bitTicks
            ? {
                afterBuildTicks: (axis: Scale) => {
                  axis.ticks = [{ value: 0 }, { value: 1 }];
                },
              }
            : {}),
        },
      },
    };
  }
}
