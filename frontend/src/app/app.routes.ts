import { inject } from '@angular/core';
import { CanActivateFn, Router, Routes } from '@angular/router';
import { Api } from './api';
import { LoginPage } from './login';
import { ProcessPage } from './process';
import { ProjectPage } from './project';

const authGuard: CanActivateFn = () =>
  inject(Api).loggedIn() || inject(Router).createUrlTree(['/login']);

export const routes: Routes = [
  { path: 'login', component: LoginPage },
  { path: 'project', component: ProjectPage, canActivate: [authGuard] },
  { path: 'process', component: ProcessPage, canActivate: [authGuard] },
  {
    // Lazy: Chart.js is only downloaded when a report is opened.
    path: 'report/:id',
    canActivate: [authGuard],
    loadComponent: () => import('./report').then((m) => m.ReportPage),
  },
  { path: '**', redirectTo: 'project' },
];
