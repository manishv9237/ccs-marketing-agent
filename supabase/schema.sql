-- Run once in Supabase: SQL Editor -> New query -> paste -> Run.
create table if not exists public.workspace_files (
  username   text not null,
  path       text not null,
  data_b64   text not null,
  updated_at timestamptz not null default now(),
  primary key (username, path)
);
-- Only the app's service key may read/write: RLS on, and no policies for anon/authenticated users.
alter table public.workspace_files enable row level security;
revoke all on public.workspace_files from anon, authenticated;
