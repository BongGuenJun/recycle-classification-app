import { Pipeline, PredictionResponse } from './types';
const base = (value: string) => value.trim().replace(/\/$/, '');
export async function predictImage(serverUrl: string, uri: string, pipeline:Pipeline='one_stage'): Promise<PredictionResponse> {
  const requestId=`${Date.now().toString(36)}-${Math.random().toString(36).slice(2,6)}`;
  const form = new FormData();
  form.append('image', { uri, name:'capture.jpg', type:'image/jpeg' } as unknown as Blob);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 30_000);
  try {
    if(__DEV__)console.info(`[predict:${requestId}] upload start`,{pipeline,serverUrl:base(serverUrl)});
    const response = await fetch(`${base(serverUrl)}/predict?pipeline=${pipeline}&confidence_threshold=0.35`, { method:'POST', body:form, signal:controller.signal });
    if(__DEV__)console.info(`[predict:${requestId}] response`,{status:response.status});
    if (!response.ok) throw new Error(response.status>=500?'서버에서 이미지를 분석하지 못했어요.':'이미지를 분석할 수 없어요. 다시 촬영해주세요.');
    const json = await response.json();
    return { ...json, detections: Array.isArray(json.detections) ? json.detections : [] };
  } catch (error) {
    if(__DEV__)console.warn(`[predict:${requestId}] failed`,{pipeline,name:error instanceof Error?error.name:'Unknown',message:error instanceof Error?error.message:String(error)});
    if (error instanceof Error && error.name === 'AbortError') throw new Error('분석 시간이 오래 걸리고 있어요. 다시 시도해주세요.');
    if (error instanceof TypeError || (error instanceof Error && /network request failed/i.test(error.message))) throw new Error('서버에 연결하지 못했어요. Wi-Fi와 서버 주소를 확인해주세요.');
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
