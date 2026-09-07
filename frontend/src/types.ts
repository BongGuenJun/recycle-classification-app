export type Screen = 'home' | 'camera' | 'guide' | 'history';
export type CameraMode = 'manual' | 'auto';
export type Pipeline = 'one_stage' | 'two_stage';
export type MaterialId = 'pet' | 'plastic' | 'can' | 'glass' | 'paper' | 'vinyl' | 'styrofoam' | 'food' | 'battery' | 'clothing';
export type Detection = { bbox: [number, number, number, number]; class_id?: number; class_name?: string; material?: string; dirtiness?: string; confidence: number; detector_confidence?: number; classifier_confidence?: number };
export type PredictionResponse = { detections: Detection[]; image_width?: number; image_height?: number; processing_time_ms?: number; inference_time_ms?: number; inference_ms?: number };
export type AnalysisLog = { id: string; cropUri: string; label: string; material?: string; dirtiness?: string; confidence: number; candidateCount: number; totalCandidateCount: number; createdAt: string; mode: CameraMode; pipeline: Pipeline };
