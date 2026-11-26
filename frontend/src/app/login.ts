import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { Api, errorText } from './api';

@Component({
  selector: 'app-login',
  imports: [FormsModule],
  template: `
    <section class="login">
      <div class="login-hero">
        <h1>Find the bits that switc and the bytes that count.</h1>
        <p class="lede">
           Upload CSV and fin digital and counter signal.
        </p>

        <svg class="trace" viewBox="0 0 520 300" role="img"
             aria-label="Example of a digital signal switching between 0 and 1, and a counter that steps up and wraps">
          <text x="0" y="16" class="t-id">CAN ID 1A0</text>
          <text x="96" y="16" class="t-loc">Byte B2, bit 2</text>
          <text x="520" y="16" text-anchor="end" class="t-digital">Digital</text>
          <line x1="0" y1="40" x2="520" y2="40" class="grid" />
          <line x1="0" y1="110" x2="520" y2="110" class="grid" />
          <path class="line digital" pathLength="1"
                d="M0 110 H60 V40 H150 V110 H190 V40 H300 V110 H340 V40 H400 V110 H520" />

          <text x="0" y="176" class="t-id">CAN ID 2B0</text>
          <text x="96" y="176" class="t-loc">Byte B1</text>
          <text x="520" y="176" text-anchor="end" class="t-counter">Counter</text>
          <line x1="0" y1="196" x2="520" y2="196" class="grid" />
          <line x1="0" y1="286" x2="520" y2="286" class="grid" />
          <path class="line counter" pathLength="1"
                d="M0 286 H36 V268 H72 V250 H108 V232 H144 V214 H180 V286 H216 V268 H252 V250 H288 V232 H324 V214 H360 V286 H396 V268 H432 V250 H468 V232 H504 V214 H520" />
        </svg>
      </div>

      <form class="login-form" (ngSubmit)="submit()" #f="ngForm">
        <h2>{{ mode() === 'login' ? 'Sign in' : 'Create an account' }}</h2>

        <div class="field">
          <label for="username">Username</label>
          <input id="username" name="username" type="text" autocomplete="username"
                 required minlength="3" maxlength="50" [(ngModel)]="username" />
        </div>
        <div class="field">
          <label for="password">Password</label>
          <input id="password" name="password" type="password" required minlength="6" maxlength="72"
                 [attr.autocomplete]="mode() === 'login' ? 'current-password' : 'new-password'"
                 [(ngModel)]="password" />
          @if (mode() === 'register') {
            <span class="hint">At least 6 characters.</span>
          }
        </div>

        @if (error()) {
          <p class="alert" role="alert">{{ error() }}</p>
        }

        <button class="btn" type="submit" [disabled]="f.invalid || busy()">
          {{ busy() ? 'Please wait…' : mode() === 'login' ? 'Sign in' : 'Create account' }}
        </button>

        <p class="switch">
          {{ mode() === 'login' ? 'New here?' : 'Already have an account?' }}
          <button type="button" class="link" (click)="toggle()">
            {{ mode() === 'login' ? 'Create an account' : 'Sign in' }}
          </button>
        </p>
      </form>
    </section>
  `,
})
export class LoginPage {
  private readonly api = inject(Api);
  private readonly router = inject(Router);

  protected readonly mode = signal<'login' | 'register'>('login');
  protected readonly username = signal('');
  protected readonly password = signal('');
  protected readonly busy = signal(false);
  protected readonly error = signal('');

  protected toggle(): void {
    this.mode.set(this.mode() === 'login' ? 'register' : 'login');
    this.error.set('');
  }

  protected submit(): void {
    this.busy.set(true);
    this.error.set('');
    this.api.login(this.mode(), this.username().trim(), this.password()).subscribe({
      next: () => this.router.navigate(['/project']),
      error: (e) => {
        this.error.set(errorText(e));
        this.busy.set(false);
      },
    });
  }
}
