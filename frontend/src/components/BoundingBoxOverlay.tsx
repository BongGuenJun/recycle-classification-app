import { StyleSheet, Text, View } from 'react-native';
import { Detection } from '../types';
import { colors } from '../theme';
import { getResultCopy } from '../resultCopy';

type Props = { detections: Detection[]; imageWidth: number; imageHeight: number; viewWidth: number; viewHeight: number };
export function BoundingBoxOverlay({ detections, imageWidth, imageHeight, viewWidth, viewHeight }: Props) {
  if (!imageWidth || !imageHeight || !viewWidth || !viewHeight) return null;
  const scale = Math.min(viewWidth / imageWidth, viewHeight / imageHeight);
  const offsetX = (viewWidth - imageWidth * scale) / 2;
  const offsetY = (viewHeight - imageHeight * scale) / 2;
  return <View pointerEvents="none" style={StyleSheet.absoluteFill}>
    {detections.map((item, index) => {
      const [x1,y1,x2,y2] = item.bbox;
      const copy=getResultCopy(item);
      return <View key={`${index}-${x1}`} style={[styles.box,{ left:offsetX+x1*scale, top:offsetY+y1*scale, width:(x2-x1)*scale, height:(y2-y1)*scale }]}>
        <Text style={styles.label}>{copy.materialName} · {copy.conditionName}</Text>
      </View>;
    })}
  </View>;
}
const styles = StyleSheet.create({
  box:{ position:'absolute', borderWidth:3, borderColor:'#66F2A3', borderRadius:5 },
  label:{ alignSelf:'flex-start', marginTop:-25, marginLeft:-3, paddingHorizontal:6, paddingVertical:3, color:colors.black, backgroundColor:'#66F2A3', fontWeight:'800', fontSize:12 },
});
