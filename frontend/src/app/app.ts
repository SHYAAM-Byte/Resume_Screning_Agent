import { Component, inject } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { Api } from './api';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    @if (api.loggedIn()) {
      <header class="topbar">
        <a class="brand" routerLink="/project">Vehicle signals</a>
        <nav aria-label="Steps">
          <ol class="steps">
            <li><a routerLink="/project" routerLinkActive="on">Project details</a></li>
            <li><a routerLink="/process" routerLinkActive="on">CSV</a></li>
            <li>
              @if (api.report(); as r) {
                <a [routerLink]="['/report', r.project.id]" routerLinkActive="on">Report</a>
              } @else {
                <span class="off">Report</span>
              }
            </li>
          </ol>
        </nav>
        <div class="who">
          <span>{{ api.username() }}</span>
          <button type="button" class="link" (click)="api.logout()">Sign out</button>
        </div>
      </header>
    }
    <main class="shell">
      <router-outlet />
    </main>
  `,
})
export class App {
  protected readonly api = inject(Api);
}
