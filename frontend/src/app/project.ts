import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Api } from './api';

@Component({
  selector: 'app-project',
  imports: [FormsModule],
  template: `
    <section class="page narrow">
      <h1>Project details</h1>
      <p class="lede">Tell us which vehicle the log comes from. This is saved with the report.</p>

      <form class="stack" (ngSubmit)="next()" #f="ngForm">
        <div class="field">
          <label for="name">Project name</label>
          <input id="name" name="name" type="text" required maxlength="100" [(ngModel)]="name" />
        </div>
        <div class="field">
          <label for="make">Vehicle make</label>
          <input id="make" name="make" type="text" required maxlength="50" [(ngModel)]="make" />
        </div>
        <div class="field">
          <label for="model">Model</label>
          <input id="model" name="model" type="text" required maxlength="50" [(ngModel)]="model" />
        </div>
        <div class="field short">
          <label for="year">Year</label>
          <input id="year" name="year" type="number" required min="1886" max="2100" step="1"
                 inputmode="numeric" [(ngModel)]="year" />
        </div>

        <div class="actions">
          <button class="btn" type="submit" [disabled]="f.invalid">Continue to CSV</button>
        </div>
      </form>
    </section>
  `,
})
export class ProjectPage {
  private readonly api = inject(Api);
  private readonly router = inject(Router);

  private readonly saved = this.api.details();
  protected readonly name = signal(this.saved?.name ?? '');
  protected readonly make = signal(this.saved?.make ?? '');
  protected readonly model = signal(this.saved?.model ?? '');
  protected readonly year = signal<number | null>(this.saved?.year ?? null);

  protected next(): void {
    const year = this.year();
    if (year === null) return;
    this.api.details.set({
      name: this.name().trim(),
      make: this.make().trim(),
      model: this.model().trim(),
      year,
    });
    this.router.navigate(['/process']);
  }
}
