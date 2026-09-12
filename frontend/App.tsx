import { StatusBar } from 'expo-status-bar';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useEffect, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { CameraTestScreen } from './src/screens/CameraTestScreen';
import { GuideScreen } from './src/screens/GuideScreen';
import { HomeScreen } from './src/screens/HomeScreen';
import { HistoryScreen } from './src/screens/HistoryScreen';
import { AnalysisLog, CameraMode, MaterialId, Pipeline, Screen } from './src/types';
import { resetSessionFiles } from './src/sessionFiles';

const SERVER_URL = process.env.EXPO_PUBLIC_API_URL;

export default function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [cameraMode, setCameraMode] = useState<CameraMode>('manual');
  const [selectedMaterial, setSelectedMaterial] = useState<MaterialId>('pet');
  const [logs, setLogs] = useState<AnalysisLog[]>([]);
  const [pipeline, setPipeline] = useState<Pipeline>('one_stage');

  useEffect(() => {
    resetSessionFiles();
    AsyncStorage.getItem('recycle.pipeline').then((saved) => {
      if (saved==='one_stage'||saved==='two_stage') setPipeline(saved);
    }).catch(() => undefined);
  }, []);

  const changePipeline=(value:Pipeline)=>{setPipeline(value);void AsyncStorage.setItem('recycle.pipeline',value);};

  const openCamera = (mode: CameraMode) => { setCameraMode(mode); setScreen('camera'); };
  const openGuide = (material: MaterialId) => { setSelectedMaterial(material); setScreen('guide'); };

  return (
    <SafeAreaProvider>
      <SafeAreaView style={styles.safe}>
        <StatusBar style={screen==='home'||screen==='guide'?'dark':'light'} />
        <View style={styles.root}>
        {screen === 'home' && <HomeScreen pipeline={pipeline} onPipelineChange={changePipeline} onOpenCamera={openCamera} onOpenHistory={()=>setScreen('history')} onOpenGuide={openGuide} logs={logs} />}
        {screen === 'camera' && <CameraTestScreen mode={cameraMode} pipeline={pipeline} serverUrl={SERVER_URL} logs={logs} onAddLog={(log) => setLogs((old) => [log, ...old])} onBack={() => setScreen('home')} />}
        {screen === 'history' && <HistoryScreen logs={logs} onBack={()=>setScreen('home')}/>} 
        {screen === 'guide' && <GuideScreen initialMaterial={selectedMaterial} onBack={() => setScreen('home')} />}
        </View>
      </SafeAreaView>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: '#F4F7F2' },
  root: { flex: 1 },
});
