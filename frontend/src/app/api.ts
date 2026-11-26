import { HttpClient, HttpErrorResponse, HttpInterceptorFn } from '@angular/common/http';
import { Injectable, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { tap } from 'rxjs';
import { environment } from '../environments/environment';
/** Backend base URL. Change it here if the API runs somewhere else. */
// export const API_URL = 'http://localhost:8000/api';
export const API_URL = environment.apiUrl;

const TOKEN_KEY = 'vsp_token';
const USER_KEY = 'vsp_user';

// ---- shapes returned by the backend (same for a fresh report and for history) ----

export interface ProjectInfo {
  id: number;
  project_name: string;
  vehicle_make: string;
  vehicle_model: string;
  vehicle_year: number;
  csv_filename: string | null;
  row_count: number | null;
  created_at: string | null;
}

export interface DigitalSignal {
  signal_type: 'digital';
  can_id: string;
  byte: number; // 1..8 (B1..B8)
  bit: number; // 0..7, 0 = MSB
  transitions: number;
  abs_time: number[];
  values: number[];
}

export interface CounterSignal {
  signal_type: 'counter';
  can_id: string;
  byte: number;
  min: number;
  max: number;
  step: number;
  abs_time: number[];
  values: number[];
}

export interface Report {
  project: ProjectInfo;
  digital_signals: DigitalSignal[];
  counter_signals: CounterSignal[];
  skipped_rows?: number; // only present right after processing
}

export interface ProjectSummary extends ProjectInfo {
  digital_count: number;
  counter_count: number;
}

/** What the Project page collects; it is sent together with the CSV. */
export interface ProjectDetails {
  name: string;
  make: string;
  model: string;
  year: number;
}

interface TokenResponse {
  access_token: string;
  username: string;
}

function read(key: string): string | null {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function write(key: string, value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {
    /* storage unavailable: the session just won't survive a reload */
  }
}

@Injectable({ providedIn: 'root' })
export class Api {
  private readonly http = inject(HttpClient);
  private readonly router = inject(Router);

  readonly token = signal<string | null>(read(TOKEN_KEY));
  readonly username = signal<string | null>(read(USER_KEY));
  readonly loggedIn = computed(() => !!this.token());

  /** State shared between the pages of the flow. */
  readonly details = signal<ProjectDetails | null>(null);
  readonly report = signal<Report | null>(null);

  login(mode: 'login' | 'register', username: string, password: string) {
    return this.http
      .post<TokenResponse>(`${API_URL}/auth/${mode}`, { username, password })
      .pipe(
        tap((res) => {
          write(TOKEN_KEY, res.access_token);
          write(USER_KEY, res.username);
          this.token.set(res.access_token);
          this.username.set(res.username);
        }),
      );
  }

  logout(): void {
    write(TOKEN_KEY, null);
    write(USER_KEY, null);
    this.token.set(null);
    this.username.set(null);
    this.details.set(null);
    this.report.set(null);
    this.router.navigate(['/login']);
  }

  process(file: File, d: ProjectDetails, digital: boolean, counter: boolean) {
    const form = new FormData();
    form.append('file', file);
    form.append('project_name', d.name);
    form.append('vehicle_make', d.make);
    form.append('vehicle_model', d.model);
    form.append('vehicle_year', String(d.year));
    form.append('digital', String(digital));
    form.append('counter', String(counter));
    return this.http.post<Report>(`${API_URL}/projects/process`, form);
  }

  history() {
    return this.http.get<ProjectSummary[]>(`${API_URL}/projects`);
  }

  project(id: number) {
    return this.http.get<Report>(`${API_URL}/projects/${id}`);
  }
}

/** Adds the JWT to API calls and signs the user out when the token is rejected. */
export const authInterceptor: HttpInterceptorFn = (req, next) => {
  const api = inject(Api);
  const token = api.token();
  const isAuthCall = req.url.includes('/auth/');
  const authed = token && req.url.startsWith(API_URL) && !isAuthCall;
  const request = authed ? req.clone({ setHeaders: { Authorization: `Bearer ${token}` } }) : req;

  return next(request).pipe(
    tap({
      error: (e) => {
        if (e instanceof HttpErrorResponse && e.status === 401 && !isAuthCall) api.logout();
      },
    }),
  );
};

/** Turns any error from the backend into one readable sentence. */
export function errorText(e: unknown): string {
  if (e instanceof HttpErrorResponse) {
    if (e.status === 0) {
      return 'Cannot reach the server. Check that the backend is running on port 8000.';
    }
    const detail: unknown = e.error?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length) {
      return detail
        .map((d: { loc?: (string | number)[]; msg?: string }) => {
          const field = d.loc?.[d.loc.length - 1];
          return field !== undefined ? `${field}: ${d.msg}` : String(d.msg);
        })
        .join('; ');
    }
    return `The request failed (HTTP ${e.status}).`;
  }
  return 'Something went wrong. Try again.';
}
