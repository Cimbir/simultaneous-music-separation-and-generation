create table if not exists participants (
  id            uuid primary key default gen_random_uuid(),
  session_index int  not null,
  user_agent    text,
  consent       bool default true,
  started_at    timestamptz default now(),
  finished_at   timestamptz
);

create table if not exists responses (
  id               bigint generated always as identity primary key,
  participant_id   uuid references participants(id) on delete cascade,
  trial_number     int,
  is_catch         bool,
  clips            jsonb,
  musicality       jsonb,
  coherence        jsonb,
  replays          jsonb,
  trial_started_at timestamptz,
  submitted_at     timestamptz default now()
);

create or replace function claim_session(p_num_sessions int, p_user_agent text)
returns table(participant_id uuid, session_index int)
language plpgsql
security definer
as $$
declare
  cnt    int;
  new_id uuid;
  sidx   int;
begin
  perform pg_advisory_xact_lock(42);
  select count(*) into cnt from participants;
  sidx := cnt % p_num_sessions;
  insert into participants(session_index, user_agent)
    values (sidx, p_user_agent)
    returning id into new_id;
  return query select new_id, sidx;
end;
$$;

alter table participants enable row level security;
alter table responses    enable row level security;

drop policy if exists anon_insert_participants on participants;
create policy anon_insert_participants on participants
  for insert to anon with check (true);

drop policy if exists anon_update_participants on participants;
create policy anon_update_participants on participants
  for update to anon using (true) with check (true);

drop policy if exists anon_insert_responses on responses;
create policy anon_insert_responses on responses
  for insert to anon with check (true);

grant execute on function claim_session(int, text) to anon;
