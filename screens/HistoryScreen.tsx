import React, { useState, useEffect } from 'react';
import { View, Text, FlatList, StyleSheet, SafeAreaView, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_BASE_URL } from '../config';

export default function HistoryScreen() {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    try {
      const userId = await AsyncStorage.getItem('user_id') || 'guest';
      const response = await fetch(`${API_BASE_URL}/history/${userId}`);
      const result = await response.json();
      if (result.status === 'success') {
        setHistory(result.history);
      }
    } catch (error) {
      console.log('Error fetching history:', error);
    } finally {
      setLoading(false);
    }
  };

  const renderItem = ({ item }: { item: any }) => (
    <View style={styles.card}>
      <View style={styles.cardHeader}>
        <Text style={styles.dateText}>{new Date(item.timestamp + (item.timestamp.endsWith('Z') ? '' : 'Z')).toLocaleString()}</Text>
        <Text style={styles.caloriesText}>{item.calories_burned.toFixed(1)} kcal</Text>
      </View>
      <View style={styles.cardBody}>
        <Text style={styles.infoText}>Duration: {item.duration} min</Text>
        <Text style={styles.infoText}>MET: {item.extracted_met}</Text>
        <Text style={styles.infoText}>Temp: {item.live_temp}°C</Text>
      </View>
    </View>
  );

  return (
    <SafeAreaView style={styles.container}>
      <Text style={styles.title}>Workout History</Text>
      {loading ? (
        <ActivityIndicator size="large" color="#007BFF" />
      ) : (
        <FlatList
          data={history}
          keyExtractor={(item) => item.id.toString()}
          renderItem={renderItem}
          contentContainerStyle={styles.listContainer}
          ListEmptyComponent={<Text style={styles.emptyText}>No workout history found.</Text>}
        />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F5F9FF' },
  title: { fontSize: 24, fontWeight: 'bold', color: '#333', margin: 20 },
  listContainer: { paddingHorizontal: 20, paddingBottom: 20 },
  card: { backgroundColor: '#fff', borderRadius: 12, padding: 16, marginBottom: 16, shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 3 },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', borderBottomWidth: 1, borderBottomColor: '#eee', paddingBottom: 8, marginBottom: 8 },
  dateText: { fontSize: 14, color: '#666', fontWeight: 'bold' },
  caloriesText: { fontSize: 16, color: '#FF5733', fontWeight: 'bold' },
  cardBody: { flexDirection: 'row', justifyContent: 'space-between' },
  infoText: { fontSize: 14, color: '#555' },
  emptyText: { textAlign: 'center', color: '#999', marginTop: 40, fontSize: 16 }
});
