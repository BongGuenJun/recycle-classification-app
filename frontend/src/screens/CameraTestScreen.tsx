import { CameraView, useCameraPermissions } from 'expo-camera';
import { File } from 'expo-file-system';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { DeviceMotion } from 'expo-sensors';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ActivityIndicator, AppState, BackHandler, Image, LayoutChangeEvent, Pressable, StyleSheet, Text, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { predictImage } from '../api';
import { BoundingBoxOverlay } from '../components/BoundingBoxOverlay';
import { AnalysisLogList } from '../components/AnalysisLogList';
import { keepSessionCrop } from '../sessionFiles';
import { getResultCopy } from '../resultCopy';
import { colors } from '../theme';
import { AnalysisLog, CameraMode, Detection, Pipeline } from '../types';

type Phase='moving'|'stabilizing'|'ready'|'capturing'|'preparing'|'analyzing'|'result'|'locked'|'not_found'|'error';
type Picture={uri:string;width:number;height:number};
type Props={mode:CameraMode;pipeline:Pipeline;serverUrl:string;logs:AnalysisLog[];onAddLog:(log:AnalysisLog)=>void;onBack:()=>void};
const ACCEL_LIMIT=0.55,ROTATION_LIMIT=10,STABLE_MS=700,MAX_IMAGE_SIDE=1600,RESULT_MS=1500;

function removeFile(uri?:string){if(!uri)return;try{const file=new File(uri);if(file.exists)file.delete();}catch{/* 다음 캐시 정리에 맡김 */}}
async function prepareImage(uri:string,width:number,height:number):Promise<Picture>{
  const context=ImageManipulator.manipulate(uri);const longest=Math.max(width,height);
  if(longest>MAX_IMAGE_SIDE){if(width>=height)context.resize({width:MAX_IMAGE_SIDE});else context.resize({height:MAX_IMAGE_SIDE});}
  const rendered=await context.renderAsync();const saved=await rendered.saveAsync({compress:.72,format:SaveFormat.JPEG});return{uri:saved.uri,width:rendered.width,height:rendered.height};
}
async function createCrop(picture:Picture,detection:Detection,id:string){
  const[x1,y1,x2,y2]=detection.bbox;const left=Math.max(0,Math.floor(x1)),top=Math.max(0,Math.floor(y1));const width=Math.max(1,Math.min(picture.width,Math.ceil(x2))-left),height=Math.max(1,Math.min(picture.height,Math.ceil(y2))-top);
  const context=ImageManipulator.manipulate(picture.uri);context.crop({originX:left,originY:top,width,height});if(width>256)context.resize({width:256});const rendered=await context.renderAsync();const saved=await rendered.saveAsync({compress:.65,format:SaveFormat.JPEG});return keepSessionCrop(saved.uri,id);
}
function selectGuideCandidates(items:Detection[],picture:Picture,view:{width:number;height:number}){
  if(!view.width||!view.height)return items;
  const scale=Math.max(view.width/picture.width,view.height/picture.height),offsetX=(view.width-picture.width*scale)/2,offsetY=(view.height-picture.height*scale)/2;
  const left=view.width*.18,right=view.width*.82,top=view.height*.21,bottom=view.height*.80;
  return items.filter(item=>{
    const[x1,y1,x2,y2]=item.bbox,boxLeft=offsetX+x1*scale,boxTop=offsetY+y1*scale,boxRight=offsetX+x2*scale,boxBottom=offsetY+y2*scale;
    const centerX=(boxLeft+boxRight)/2,centerY=(boxTop+boxBottom)/2,centerInside=centerX>=left&&centerX<=right&&centerY>=top&&centerY<=bottom;
    const intersectionWidth=Math.max(0,Math.min(boxRight,right)-Math.max(boxLeft,left)),intersectionHeight=Math.max(0,Math.min(boxBottom,bottom)-Math.max(boxTop,top));
    const boxArea=Math.max(1,(boxRight-boxLeft)*(boxBottom-boxTop)),overlapRatio=(intersectionWidth*intersectionHeight)/boxArea;
    return centerInside||overlapRatio>=.30;
  });
}

export function CameraTestScreen({mode,pipeline,serverUrl,logs,onAddLog,onBack}:Props){
  const camera=useRef<CameraView>(null),busy=useRef(false),stableSince=useRef<number|null>(null),resultTimer=useRef<ReturnType<typeof setTimeout>|null>(null),photoRef=useRef<Picture|null>(null);const phaseRef=useRef<Phase>(mode==='auto'?'moving':'ready');
  const insets=useSafeAreaInsets();const[permission,requestPermission]=useCameraPermissions();const[cameraReady,setCameraReady]=useState(false);const[panelOpen,setPanelOpen]=useState(false);const[appActive,setAppActive]=useState(AppState.currentState==='active');
  const[phase,setPhaseState]=useState<Phase>(mode==='auto'?'moving':'ready');const setPhase=(p:Phase)=>{phaseRef.current=p;setPhaseState(p);};
  const[photo,setPhotoState]=useState<Picture|null>(null);const setPhoto=(p:Picture|null)=>{photoRef.current=p;setPhotoState(p);};
  const[detections,setDetections]=useState<Detection[]>([]),[error,setError]=useState('');const[metrics,setMetrics]=useState({accel:0,rotation:0,stable:0}),[view,setView]=useState({width:0,height:0});
  const clearTimer=()=>{if(resultTimer.current){clearTimeout(resultTimer.current);resultTimer.current=null;}};
  const clearResult=useCallback((nextPhase:Phase)=>{clearTimer();removeFile(photoRef.current?.uri);setPhoto(null);setDetections([]);setError('');stableSince.current=null;busy.current=false;setPhase(nextPhase);},[]);
  const nextAnalysis=useCallback(()=>clearResult(mode==='auto'?'moving':'ready'),[clearResult,mode]);
  const handleBack=useCallback(()=>{if(panelOpen){setPanelOpen(false);return;}if(photoRef.current){clearResult(mode==='auto'?'locked':'ready');return;}onBack();},[clearResult,mode,onBack,panelOpen]);
  useEffect(()=>{const sub=BackHandler.addEventListener('hardwareBackPress',()=>{handleBack();return true;});return()=>sub.remove();},[handleBack]);
  useEffect(()=>()=>{clearTimer();removeFile(photoRef.current?.uri);},[]);
  useEffect(()=>{const sub=AppState.addEventListener('change',state=>{const active=state==='active';setAppActive(active);if(!active&&mode==='auto'&&['moving','stabilizing','ready'].includes(phaseRef.current))setPhase('locked');});return()=>sub.remove();},[mode]);

  const analyze=useCallback(async()=>{
    if(busy.current||!cameraReady||!camera.current)return;busy.current=true;setError('');setPhase('capturing');let originalUri:string|undefined;
    try{
      const original=await camera.current.takePictureAsync({quality:.65,skipProcessing:false,shutterSound:false});if(!original)throw new Error('사진을 저장하지 못했습니다.');originalUri=original.uri;setPhase('preparing');
      const prepared=await prepareImage(original.uri,original.width,original.height);removeFile(original.uri);originalUri=undefined;setPhoto(prepared);setPhase('analyzing');
      const response=await predictImage(serverUrl,prepared.uri,pipeline);const all=[...(response.detections||[])].sort((a,b)=>b.confidence-a.confidence);const found=selectGuideCandidates(all,prepared,view);setDetections(found);
      if(!found.length){setPhase('not_found');return;}
      const primary=found[0],copy=getResultCopy(primary),id=`${Date.now()}`;const cropUri=await createCrop(prepared,primary,id);onAddLog({id,cropUri,label:copy.materialName,material:copy.material,dirtiness:copy.dirtiness,confidence:primary.confidence,candidateCount:found.length,totalCandidateCount:all.length,createdAt:new Date().toISOString(),mode,pipeline});setPhase('result');
      if(mode==='auto')resultTimer.current=setTimeout(()=>clearResult('locked'),RESULT_MS);
    }catch(e){if(originalUri)removeFile(originalUri);const message=e instanceof Error&&/[가-힣]/.test(e.message)?e.message:'이미지를 분석하지 못했어요. 다시 촬영해주세요.';setError(message);setPhase('error');}finally{busy.current=false;}
  },[cameraReady,clearResult,mode,onAddLog,pipeline,serverUrl,view]);

  useEffect(()=>{
    if(mode!=='auto'||photo||!cameraReady||panelOpen||phase==='locked'||!appActive)return;DeviceMotion.setUpdateInterval(200);
    const sub=DeviceMotion.addListener(data=>{if(busy.current||!['moving','stabilizing'].includes(phaseRef.current))return;const a=data.acceleration,r=data.rotationRate;const accel=Math.sqrt((a?.x||0)**2+(a?.y||0)**2+(a?.z||0)**2),rotation=Math.sqrt((r?.alpha||0)**2+(r?.beta||0)**2+(r?.gamma||0)**2),now=Date.now();if(accel<=ACCEL_LIMIT&&rotation<=ROTATION_LIMIT){if(stableSince.current===null)stableSince.current=now;const stable=now-stableSince.current;setMetrics({accel,rotation,stable});setPhase('stabilizing');if(stable>=STABLE_MS)void analyze();}else{stableSince.current=null;setMetrics({accel,rotation,stable:0});setPhase('moving');}});return()=>sub.remove();
  },[analyze,appActive,cameraReady,mode,panelOpen,phase,photo]);

  if(!permission)return <Centered text="카메라 권한을 확인하고 있습니다…"/>;
  if(!permission.granted)return <View style={styles.permission}><Text style={styles.permissionTitle}>카메라 권한이 필요해요</Text><Text style={styles.permissionText}>사진은 분석 요청에만 사용됩니다.</Text><Pressable style={styles.action} onPress={requestPermission}><Text style={styles.actionText}>권한 허용</Text></Pressable><Pressable onPress={onBack}><Text style={styles.backText}>홈으로</Text></Pressable></View>;
  const primary=detections[0],visible=primary?[primary]:[];const status=phase==='moving'?'물체 중심을 가이드에 맞춰주세요':phase==='stabilizing'||phase==='capturing'||phase==='preparing'?'대상 확인 중…':phase==='analyzing'?'AI 분석 중…':phase==='result'?'분석 완료':phase==='locked'?'다음 물체를 준비해주세요':phase==='not_found'?'확실한 재활용품을 찾지 못했어요':phase==='error'?'분석 오류':'분석 준비 완료';
  const collapsedHeight=68+insets.bottom;
  return <View style={styles.root}>
    <View style={styles.top}><Pressable onPress={handleBack}><Text style={styles.back}>‹</Text></Pressable><View><Text style={styles.topTitle}>{mode==='manual'?'수동 촬영':'자동 촬영'}</Text><Text style={styles.topSub}>{pipeline==='one_stage'?'1-stage · 빠른 통합 분석':'2-stage · 단계별 분석'}</Text></View><View style={{width:28}}/></View>
    <View style={[styles.stage,{paddingBottom:collapsedHeight}]} onLayout={(e:LayoutChangeEvent)=>setView(e.nativeEvent.layout)}>
      {!photo&&!panelOpen&&appActive?<><CameraView ref={camera} style={StyleSheet.absoluteFill} facing="back" onCameraReady={()=>setCameraReady(true)}/>{phase!=='locked'&&<><View pointerEvents="none" style={styles.guideFrame}/><Text style={styles.guideText}>재활용품 하나의 중심을 맞춰주세요</Text></>}</>:photo?<><Image source={{uri:photo.uri}} style={StyleSheet.absoluteFill} resizeMode="contain"/><BoundingBoxOverlay detections={visible} imageWidth={photo.width} imageHeight={photo.height} viewWidth={view.width} viewHeight={view.height}/></>:<View style={styles.paused}><Text style={styles.pausedText}>{appActive?'최근 분석을 확인하고 있어요':'카메라가 일시정지됐어요'}</Text></View>}
      <View style={[styles.status,phase==='result'&&styles.statusGood,phase==='not_found'&&styles.statusWarn]}><Text style={styles.statusText}>{status}</Text>{phase==='result'&&primary&&<Text style={styles.statusConfidence}>AI 확신도 {Math.round(primary.confidence*100)}%</Text>}</View>
      {['capturing','preparing','analyzing'].includes(phase)&&<View style={styles.loading}><ActivityIndicator size="large" color={colors.white}/></View>}
      {!panelOpen&&!photo&&mode==='manual'&&<Pressable style={[styles.shutter,{bottom:collapsedHeight+14},!cameraReady&&{opacity:.4}]} disabled={!cameraReady||busy.current} onPress={()=>void analyze()}><View style={styles.shutterInner}/></Pressable>}
      {!panelOpen&&((mode==='auto'&&['locked','not_found','error'].includes(phase))||(mode==='manual'&&photo))&&<Pressable style={[styles.next,{bottom:collapsedHeight+14}]} onPress={nextAnalysis}><Text style={styles.nextText}>다음 물체 분석</Text></Pressable>}
      {phase==='error'&&<Text style={[styles.error,{bottom:collapsedHeight+76}]}>{error}</Text>}
    </View>
    <ResultsSheet logs={logs} expanded={panelOpen} onExpandedChange={setPanelOpen} bottomInset={insets.bottom}/>
  </View>;
}

function ResultsSheet({logs,expanded,onExpandedChange,bottomInset}:{logs:AnalysisLog[];expanded:boolean;onExpandedChange:(v:boolean)=>void;bottomInset:number}){
  return <View style={[styles.sheet,expanded?styles.sheetOpen:{height:68+bottomInset},{paddingBottom:bottomInset}]}>
    <Pressable style={styles.sheetHeader} onPress={()=>onExpandedChange(!expanded)}><View style={styles.sheetTitleRow}><Text style={styles.sheetTitle}>최근 분석 {logs.length}건</Text><Text style={styles.sheetAction}>{expanded?'목록 닫기 ↓':'목록 보기 ↑'}</Text></View></Pressable>
    {expanded&&<AnalysisLogList logs={logs}/>} 
  </View>;
}
function Centered({text}:{text:string}){return <View style={styles.permission}><ActivityIndicator color={colors.green}/><Text style={styles.permissionText}>{text}</Text></View>}
const styles=StyleSheet.create({root:{flex:1,backgroundColor:colors.black},top:{height:70,backgroundColor:colors.black,paddingHorizontal:18,flexDirection:'row',alignItems:'center',justifyContent:'space-between'},back:{color:colors.white,fontSize:36},topTitle:{color:colors.white,fontSize:17,fontWeight:'900',textAlign:'center'},topSub:{color:'#A8B3AD',fontSize:11,textAlign:'center',marginTop:3},stage:{flex:1,backgroundColor:'#050706',overflow:'hidden'},guideFrame:{position:'absolute',left:'18%',right:'18%',top:'21%',bottom:'20%',borderWidth:2,borderColor:'rgba(255,255,255,.75)',borderRadius:22},guideText:{position:'absolute',bottom:95,alignSelf:'center',color:colors.white,fontSize:11,backgroundColor:'rgba(0,0,0,.55)',paddingHorizontal:10,paddingVertical:6,borderRadius:12},status:{position:'absolute',top:14,alignSelf:'center',backgroundColor:'rgba(0,0,0,.72)',borderRadius:18,paddingHorizontal:15,paddingVertical:9,alignItems:'center'},statusGood:{backgroundColor:'rgba(31,95,68,.92)'},statusWarn:{backgroundColor:'rgba(150,97,25,.92)'},statusText:{color:colors.white,fontWeight:'800'},statusConfidence:{color:'#DDF5E8',fontSize:11,fontWeight:'800',marginTop:3},loading:{...StyleSheet.absoluteFillObject,alignItems:'center',justifyContent:'center',backgroundColor:'rgba(0,0,0,.3)'},paused:{...StyleSheet.absoluteFillObject,alignItems:'center',justifyContent:'center',backgroundColor:'#17201C'},pausedText:{color:'#AFC0B7'},shutter:{position:'absolute',width:72,height:72,borderRadius:36,borderColor:colors.white,borderWidth:4,alignSelf:'center',left:'50%',marginLeft:-36,padding:5},shutterInner:{flex:1,borderRadius:30,backgroundColor:colors.green},next:{position:'absolute',left:42,right:42,backgroundColor:colors.green,borderRadius:14,padding:14,alignItems:'center'},nextText:{color:colors.white,fontWeight:'900'},error:{position:'absolute',left:24,right:24,color:'#FFD2D0',backgroundColor:'rgba(80,0,0,.72)',padding:10,borderRadius:10,textAlign:'center'},sheet:{position:'absolute',left:0,right:0,bottom:0,backgroundColor:colors.pale,borderTopLeftRadius:26,borderTopRightRadius:26,overflow:'hidden'},sheetClosed:{height:68},sheetOpen:{height:'64%'},sheetHeader:{height:68,paddingHorizontal:22,justifyContent:'center'},sheetTitleRow:{flexDirection:'row',justifyContent:'space-between',alignItems:'center'},sheetTitle:{fontSize:16,fontWeight:'900',color:colors.ink},sheetAction:{fontSize:12,color:colors.green,fontWeight:'900'},permission:{flex:1,alignItems:'center',justifyContent:'center',backgroundColor:colors.pale,padding:28},permissionTitle:{fontSize:23,fontWeight:'900',color:colors.ink},permissionText:{color:colors.muted,marginTop:10,textAlign:'center'},action:{backgroundColor:colors.green,paddingHorizontal:30,paddingVertical:14,borderRadius:13,marginTop:22},actionText:{color:colors.white,fontWeight:'900'},backText:{color:colors.green,marginTop:20,fontWeight:'800'}});
