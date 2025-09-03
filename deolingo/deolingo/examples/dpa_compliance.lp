% Deolingo program to check DPA text against Requirement rules
% ==========================================================

% --- Requirement Definitions ---
% Assign IDs and represent the core content of each requirement rule.

requirement(req1). % Requirement 1 ID
req_modality(req1, obligatory).
req_action(req1, make_available(processor, information)).
req_condition(req1, necessary_for_compliance(information, article_28)).

requirement(req2). % Requirement 2 ID
req_modality(req2, obligatory).
req_action(req2, allow_and_contribute(processor, audits)).
req_condition(req2, conducted_by(audits, controller)).

% --- DPA Segment Content Representation ---
% Represent what the DPA explicitly states using these facts.

dpa_states(dpa1, obligatory, make_available(processor, information)).
dpa_condition(dpa1, necessary_for_compliance(information, dpa_obligations)).

dpa_states(dpa2, obligatory, allow_and_contribute(processor, audits)).
dpa_condition(dpa2, conducted_by(audits, controller)).
dpa_condition(dpa2, conducted_by(audits, authorized_third_party)).

% --- Matching and Status Logic ---

% Helper: Does the DPA mention a specific action with any modality?
dpa_mentions_action(Action) :- dpa_states(_, _, Action).

% Status Rule 1: Satisfaction
% A requirement R is satisfied if the DPA states the same modality M for the same action A.
satisfies(R) :-
    requirement(R),
    req_modality(R, M),
    req_action(R, A),
    dpa_states(_, M, A). % Check if DPA states the exact same modality and action

% Status Rule 2: Violation
% A requirement R is violated if the DPA states a conflicting modality for the same action A.
violates(R) :-
    requirement(R),
    req_modality(R, obligatory), % Requirement is Obligatory
    req_action(R, A),
    dpa_states(_, forbidden, A).   % DPA Forbids it

% Status Rule 3: Non-Mention
% A requirement R is not mentioned if it's neither satisfied nor violated by the DPA text.
not_mentioned(R) :-
    requirement(R),
    not satisfies(R),
    not violates(R).

% --- Show Directives ---
#show satisfies/1.
#show violates/1.
#show not_mentioned/1.