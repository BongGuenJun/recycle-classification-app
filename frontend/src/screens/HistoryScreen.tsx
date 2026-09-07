import { BackHandler, Pressable, StyleSheet, Text, View } from 'react-native';
import { useEffect } from 'react';
import { AnalysisLogList } from '../components/AnalysisLogList';
import { colors } from '../theme';
import { AnalysisLog } from '../types';

export function HistoryScreen({logs,onBack}:{logs:AnalysisLog[];onBack:()=>void}){
  useEffect(()=>{const sub=BackHandler.addEventListener('hardwareBackPress',()=>{onBack();return true;});return()=>sub.remove();},[onBack]);
  return <View style={styles.root}><View style={styles.top}><Pressable onPress={onBack}><Text style={styles.back}>‹</Text></Pressable><View><Text style={styles.title}>최근 분석</Text><Text style={styles.sub}>현재 실행에서 분석한 {logs.length}건</Text></View><View style={{width:28}}/></View><AnalysisLogList logs={logs}/></View>;
}
const styles=StyleSheet.create({root:{flex:1,backgroundColor:colors.pale},top:{height:70,paddingHorizontal:18,flexDirection:'row',alignItems:'center',justifyContent:'space-between',marginBottom:12},back:{color:colors.ink,fontSize:36},title:{color:colors.ink,fontSize:19,fontWeight:'900',textAlign:'center'},sub:{color:colors.muted,fontSize:11,textAlign:'center',marginTop:3}});
