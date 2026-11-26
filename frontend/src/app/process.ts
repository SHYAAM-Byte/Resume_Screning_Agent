import { DatePipe } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { Api, ProjectSummary, errorText } from './api';

@Component({
  selector: 'app-process',
  imports: [FormsModule, RouterLink, DatePipe],
  template: `
    <section class="page">
      @if (details(); as d) {
        <h1>Process a CSV</h1>
        <p class="lede">
          {{ d.name }}: {{ d.year }} {{ d.make }} {{ d.model }}.
          <a routerLink="/project">Edit details</a>
        </p>

        <div class="process-grid">
          <div class="stack">
            <input id="csv" class="sr-only" type="file" accept=".csv,text/csv"
                   (change)="onPick($event)" />
            <label for="csv" class="drop" [class.over]="dragging()"
                   (dragover)="onDragOver($event)" (dragleave)="dragging.set(false)"
                   (drop)="onDrop($event)">
              @if (file(); as f) {
                <strong>{{ f.name }}</strong>
                <span>{{ size() }}. Click or drop to choose a different file.</span>
              } @else {
                <strong>Choose a CSV file</strong>
                <span>Click to browse, or drop a file here. Columns: abs_time, id, B1 to B8 (hex).</span>
              }
            </label>

            <fieldset class="checks">
              <legend>Detect</legend>
              <label class="check">
                <input type="checkbox" [(ngModel)]="digital" name="digital" />
                <span>
                  <strong>Digital signals</strong>
                  <small>Bits that switch repeatedly between 0 and 1</small>
                </span>
              </label>
              <label class="check">
                <input type="checkbox" [(ngModel)]="counter" name="counter" />
                <span>
                  <strong>Counter signals</strong>
                  <small>Bytes that count up by a steady step and wrap around</small>
                </span>
              </label>
            </fieldset>

            @if (error()) {
              <p class="alert" role="alert">{{ error() }}</p>
            }

            <div class="actions">
              <button class="btn" type="button" (click)="generate()" [disabled]="!canGenerate()">
                {{ busy() ? 'Processing…' : 'Generate report' }}
              </button>
              <button class="btn ghost" type="button" (click)="toggleHistory()"
                      [attr.aria-expanded]="showHistory()">
                View history
              </button>
            </div>
          </div>
        </div>

        @if (showHistory()) {
          <section class="history" aria-label="Previous projects">
            <h2>Previous projects</h2>
            @if (historyError()) {
              <p class="alert" role="alert">{{ historyError() }}</p>
            } @else if (history() === null) {
              <p class="muted">Loading…</p>
            } @else if (history()!.length === 0) {
              <p class="muted">
                No saved projects yet. Generate a report and it will appear here.
              </p>
            } @else {
              <div class="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Project</th>
                      <th>Vehicle</th>
                      <th>Processed</th>
                      <th>Digital</th>
                      <th>Counter</th>
                      <th><span class="sr-only">Open</span></th>
                    </tr>
                  </thead>
                  <tbody>
                    @for (h of history(); track h.id) {
                      <tr>
                        <td><strong>{{ h.project_name }}</strong></td>
                        <td>{{ h.vehicle_year }} {{ h.vehicle_make }} {{ h.vehicle_model }}</td>
                        <td>{{ h.created_at | date: 'medium' }}</td>
                        <td class="num">{{ h.digital_count }}</td>
                        <td class="num">{{ h.counter_count }}</td>
                        <td>
                          <a class="btn ghost small" [routerLink]="['/report', h.id]">Open report</a>
                        </td>
                      </tr>
                    }
                  </tbody>
                </table>
              </div>
            }
          </section>
        }
      }
    </section>
  `,
})
export class ProcessPage {
  private readonly api = inject(Api);
  private readonly router = inject(Router);

  protected readonly details = this.api.details;
  protected readonly file = signal<File | null>(null);
  protected readonly digital = signal(true);
  protected readonly counter = signal(true);
  protected readonly busy = signal(false);
  protected readonly error = signal('');
  protected readonly dragging = signal(false);

  protected readonly showHistory = signal(false);
  protected readonly history = signal<ProjectSummary[] | null>(null);
  protected readonly historyError = signal('');

  protected readonly canGenerate = computed(
    () => !!this.file() && (this.digital() || this.counter()) && !this.busy(),
  );
  protected readonly size = computed(() => {
    const bytes = this.file()?.size ?? 0;
    return bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(1)} KB`
      : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  });

  constructor() {
    // The details live in memory only, so a page reload sends the user back to step 1.
    if (!this.api.details()) this.router.navigate(['/project']);
  }

  protected onPick(event: Event): void {
    this.setFile((event.target as HTMLInputElement).files?.[0] ?? null);
  }

  protected onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(true);
  }

  protected onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragging.set(false);
    this.setFile(event.dataTransfer?.files?.[0] ?? null);
  }

  private setFile(f: File | null): void {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith('.csv')) {
      this.error.set('Choose a file that ends in .csv.');
      return;
    }
    this.error.set('');
    this.file.set(f);
  }

  protected generate(): void {
    const file = this.file();
    const details = this.api.details();
    if (!file || !details) return;
    this.busy.set(true);
    this.error.set('');
    this.api.process(file, details, this.digital(), this.counter()).subscribe({
      next: (report) => {
        this.api.report.set(report);
        this.router.navigate(['/report', report.project.id]);
      },
      error: (e) => {
        this.error.set(errorText(e));
        this.busy.set(false);
      },
    });
  }

  protected toggleHistory(): void {
    const open = !this.showHistory();
    this.showHistory.set(open);
    if (!open) return;
    this.history.set(null);
    this.historyError.set('');
    this.api.history().subscribe({
      next: (rows) => this.history.set(rows),
      error: (e) => this.historyError.set(errorText(e)),
    });
  }
}
