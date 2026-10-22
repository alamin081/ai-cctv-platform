import json
import time
import urllib.request

def test_pipeline_job():
    print("Testing /api/jobs ...")
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/jobs',
        data=json.dumps({'video_filename': 'test.mp4'}).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        job_id = res['job']['job_id']
        print(f"Created job: {job_id}")

    for i in range(40):
        time.sleep(0.5)
        with urllib.request.urlopen(f'http://127.0.0.1:8000/api/jobs/{job_id}') as s_resp:
            job = json.loads(s_resp.read().decode())
            status = job['status']
            prog = job['progress']
            cur_f = job['current_frame']
            tot_f = job['total_frames']
            fps = job['pipeline_fps']
            print(f"[{i*0.5:.1f}s] Status: {status} | Frame: {cur_f}/{tot_f} ({prog:.1f}%) | Speed: {fps:.1f} FPS")

            if status in ('COMPLETED', 'FAILED'):
                print("Job finished with status:", status)
                if status == 'FAILED':
                    print("Error message:", job.get('error_message'))
                else:
                    print("Has annotated video:", job.get('has_annotated_video'))
                    print("Summary:", json.dumps(job.get('final_summary'), indent=2))
                break

if __name__ == '__main__':
    test_pipeline_job()
