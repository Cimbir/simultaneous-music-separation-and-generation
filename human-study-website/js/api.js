export function createStudyApi(config) {
  const restUrl = `${config.SUPABASE_URL}/rest/v1`;

  function authHeaders(extra = {}) {
    return {
      apikey: config.SUPABASE_ANON_KEY,
      Authorization: `Bearer ${config.SUPABASE_ANON_KEY}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
      ...extra,
    };
  }

  async function request(path, options = {}) {
    const res = await fetch(`${restUrl}/${path}`, {
      ...options,
      headers: authHeaders(options.headers),
    });
    if (!res.ok) throw new Error(`Supabase ${res.status}: ${await res.text()}`);
    if (res.status === 201 || res.status === 204) return null;
    const body = await res.text();
    return body ? JSON.parse(body) : null;
  }

  return {
    async claimSession(numSessions) {
      const rows = await request("rpc/claim_session", {
        method: "POST",
        headers: { Prefer: "return=representation" },
        body: JSON.stringify({
          p_num_sessions: numSessions,
          p_user_agent: navigator.userAgent.slice(0, 300),
        }),
      });
      const row = Array.isArray(rows) ? rows[0] : rows;
      return { participantId: row.participant_id, sessionIndex: row.session_index };
    },

    saveResponse(response) {
      return request("responses", { method: "POST", body: JSON.stringify(response) });
    },

    markParticipantFinished(participantId) {
      return request(`participants?id=eq.${participantId}`, {
        method: "PATCH",
        body: JSON.stringify({ finished_at: new Date().toISOString() }),
      });
    },
  };
}
