import { useState } from 'react';
import { FlatList, Image, Pressable, StyleSheet, Text, View } from 'react-native';
import { getResultCopy } from '../resultCopy';
import { colors } from '../theme';
import { AnalysisLog } from '../types';

export function AnalysisLogList({logs}:{logs:AnalysisLog[]}){
  const[openLog,setOpenLog]=useState<string|null>(null);
  return <FlatList
    style={styles.list}
    contentContainerStyle={styles.content}
    data={logs}
    keyExtractor={item=>item.id}
    renderItem={({item})=>{
      const copy=getResultCopy(item),isOpen=openLog===item.id;
      return <Pressable style={styles.card} onPress={()=>setOpenLog(isOpen?null:item.id)}>
        <View style={styles.summary}><Image source={{uri:item.cropUri}} style={styles.crop}/><View style={{flex:1}}>
          <Text style={styles.label}>{copy.materialName}</Text><Text style={styles.action}>{copy.action}</Text>
          <Text style={[styles.meta,item.confidence<.4&&styles.uncertain]}>AI 확신도 {Math.round(item.confidence*100)}%{item.confidence<.4?' · 결과가 불확실해요':''}</Text>
          <Text style={styles.time}>{new Date(item.createdAt).toLocaleTimeString('ko-KR',{hour:'2-digit',minute:'2-digit'})}</Text>
        </View><Text style={styles.chevron}>{isOpen?'⌃':'⌄'}</Text></View>
        {isOpen&&<View style={styles.detail}><Text style={styles.detailTitle}>분석 결과</Text>
          <Text style={styles.detailLine}>분석 모델: {item.pipeline==='one_stage'?'1-stage':'2-stage'}</Text><Text style={styles.detailLine}>재질: {copy.materialName}</Text><Text style={styles.detailLine}>상태: {copy.conditionName}</Text><Text style={styles.detailLine}>AI 확신도: {Math.round(item.confidence*100)}%</Text><Text style={styles.detailLine}>가이드 내 후보: {item.candidateCount}개 · 전체 후보: {item.totalCandidateCount}개</Text>
          <Text style={styles.detailTitle}>{copy.materialName} 배출 방법</Text>{copy.steps.map((step,index)=><Text style={styles.step} key={step}>{index+1}. {step}</Text>)}<Text style={styles.disclaimer}>AI 분석 결과이며 실제 배출 기준은 지역별로 다를 수 있어요.</Text>
        </View>}
      </Pressable>;
    }}
    ListEmptyComponent={<Text style={styles.empty}>아직 분석 기록이 없어요.</Text>}
  />;
}
const styles=StyleSheet.create({list:{flex:1},content:{paddingHorizontal:18,paddingBottom:25},empty:{color:colors.muted,textAlign:'center',marginTop:50},card:{backgroundColor:colors.white,borderRadius:16,padding:11,marginBottom:10},summary:{flexDirection:'row',gap:13,alignItems:'center'},crop:{width:82,height:82,borderRadius:11,backgroundColor:colors.line},label:{fontSize:17,fontWeight:'900',color:colors.ink},action:{color:colors.ink,fontSize:12,marginTop:4},meta:{color:colors.green,marginTop:5,fontWeight:'800',fontSize:12},uncertain:{color:'#A56B13'},time:{color:colors.muted,fontSize:11,marginTop:3},chevron:{fontSize:18,color:colors.green},detail:{borderTopWidth:1,borderTopColor:colors.line,marginTop:12,paddingTop:12},detailTitle:{fontWeight:'900',color:colors.ink,marginTop:3,marginBottom:7},detailLine:{color:colors.muted,fontSize:12,lineHeight:19},step:{color:colors.ink,fontSize:12,lineHeight:20},disclaimer:{color:colors.muted,fontSize:10,lineHeight:16,marginTop:10}});
