import json
import sys

# Read candidates and look for honeypot-like patterns
with open(sys.argv[1], 'r', encoding='utf-8') as f:
    suspicious = []
    for i, line in enumerate(f):
        c = json.loads(line)
        skills = c.get('skills', [])
        
        # Check for expert-with-zero-duration pattern
        expert_zero = sum(1 for s in skills if s.get('proficiency') == 'expert' and s.get('duration_months', 0) == 0)
        
        # Check for many expert skills
        expert_count = sum(1 for s in skills if s.get('proficiency') == 'expert')
        
        n_skills = len(skills)
        avg_endorse = sum(s.get('endorsements', 0) for s in skills) / max(n_skills, 1)
        
        # Check career mismatch
        profile = c.get('profile', {})
        yoe = profile.get('years_of_experience', 0)
        career = c.get('career_history', [])
        total_career = sum(j.get('duration_months', 0) for j in career) / 12
        
        # Check timeline issues
        timeline_issue = False
        for job in career:
            start = job.get('start_date', '')
            end = job.get('end_date')
            dur = job.get('duration_months', 0)
            if start and end:
                from datetime import datetime
                try:
                    s = datetime.strptime(start, '%Y-%m-%d')
                    e = datetime.strptime(end, '%Y-%m-%d')
                    actual = (e.year - s.year) * 12 + (e.month - s.month)
                    if abs(actual - dur) > 12:
                        timeline_issue = True
                except:
                    pass
        
        # Title-description mismatch
        title = profile.get('current_title', '').lower()
        desc_mismatch = False
        if career:
            desc = career[0].get('description', '').lower()
            tech_titles = ['ai', 'ml', 'machine learning', 'data scien', 'engineer']
            nontech_desc = ['marketing', 'accounting', 'hr', 'supply chain', 'brand', 'sales', 'support agents']
            if any(t in title for t in tech_titles):
                if any(t in desc for t in nontech_desc) and not any(t in desc for t in ['ml', 'ai', 'model', 'neural', 'embedding']):
                    desc_mismatch = True
        
        # Flag suspicious
        score = 0
        if expert_zero >= 2: score += 2
        if expert_count >= 7: score += 2
        if yoe > 0 and total_career > 0 and yoe / total_career > 2.5: score += 1.5
        if timeline_issue: score += 1.5
        if desc_mismatch: score += 1.5
        
        if score >= 2:
            cid = c['candidate_id']
            suspicious.append({
                'id': cid,
                'title': profile.get('current_title', ''),
                'yoe': yoe,
                'career_yrs': round(total_career, 1),
                'expert_zero': expert_zero,
                'expert_count': expert_count,
                'n_skills': n_skills,
                'timeline': timeline_issue,
                'desc_mismatch': desc_mismatch,
                'score': score
            })

print(f"Total candidates scanned: {i+1}")
print(f"Suspicious (score >= 2): {len(suspicious)}")
print()
for s in sorted(suspicious, key=lambda x: -x['score'])[:20]:
    print(f"  {s['id']}: title={s['title']}, yoe={s['yoe']}, career={s['career_yrs']}y, "
          f"expert_zero={s['expert_zero']}, expert_count={s['expert_count']}, "
          f"timeline={s['timeline']}, desc_mismatch={s['desc_mismatch']}, score={s['score']}")
